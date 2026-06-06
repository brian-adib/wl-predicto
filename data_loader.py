import pandas as pd
import numpy as np
import os

def load_results(filepath=None):
    if filepath is None:
        if os.path.exists("data/processed/results_clean.csv"):
            filepath = "data/processed/results_clean.csv"
        else:
            filepath = "data/raw/results.csv"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró {filepath}")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

def load_elo_ratings(filepath=None):
    if filepath is None:
        if os.path.exists("data/processed/eloratings_clean.csv"):
            filepath = "data/processed/eloratings_clean.csv"
        else:
            filepath = "data/raw/eloratings.csv"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró {filepath}")
    elo_df = pd.read_csv(filepath)
    elo_df['date'] = pd.to_datetime(elo_df['date'], errors='coerce')
    elo_df = elo_df.dropna(subset=['date'])
    elo_df = elo_df.sort_values('date')
    if 'rating' in elo_df.columns:
        elo_df = elo_df.rename(columns={'rating': 'elo'})
    elo_df['elo'] = pd.to_numeric(elo_df['elo'], errors='coerce')
    return elo_df

def load_market_values(filepath="data/processed/team_market_values.csv"):
    """Carga el diccionario de valores de mercado de 2026 (fallback)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró {filepath}. Ejecuta extract_market_values.py primero.")
    df = pd.read_csv(filepath)
    if 'team' not in df.columns or 'market_value_eur' not in df.columns:
        raise ValueError("El archivo debe tener columnas 'team' y 'market_value_eur'")
    return dict(zip(df['team'], df['market_value_eur']))

def load_historical_market_values(filepath="data/processed/historical_team_market_values.csv"):
    """Carga el CSV histórico (team, date, market_value_in_eur) si existe."""
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['team', 'date'])
    return df

def get_historical_market_value(team, match_date, market_df, default=50_000_000):
    """Devuelve el valor de mercado histórico (millones) o el default."""
    if market_df is None:
        return default / 1_000_000
    team_data = market_df[market_df['team'] == team]
    if team_data.empty:
        return default / 1_000_000
    prev = team_data[team_data['date'] < match_date]
    if not prev.empty:
        return prev.iloc[-1]['market_value_in_eur'] / 1_000_000
    else:
        return team_data.iloc[0]['market_value_in_eur'] / 1_000_000

# Compatibilidad: get_market_value para usar con diccionario
def get_market_value(team, market_dict, default=50_000_000):
    return market_dict.get(team, default) / 1_000_000

def get_last_elo_before_date(team, date, elo_df, default=1500):
    hist = elo_df[(elo_df['team'] == team) & (elo_df['date'] < date)]
    if not hist.empty:
        return hist.iloc[-1]['elo']
    return default

def get_team_avg_goals(team, date, df, as_home=True, window=5):
    if as_home:
        matches = df[(df['home_team'] == team) & (df['date'] < date)].sort_values('date').tail(window)
        goals = matches['home_score']
    else:
        matches = df[(df['away_team'] == team) & (df['date'] < date)].sort_values('date').tail(window)
        goals = matches['away_score']
    if not goals.empty:
        return goals.mean()
    return 1.0

def get_team_form(team, date, df, window=5):
    home_matches = df[(df['home_team'] == team) & (df['date'] < date)].sort_values('date').tail(window)
    away_matches = df[(df['away_team'] == team) & (df['date'] < date)].sort_values('date').tail(window)
    all_matches = pd.concat([
        home_matches[['date', 'home_team', 'away_team', 'home_score', 'away_score']],
        away_matches[['date', 'home_team', 'away_team', 'home_score', 'away_score']]
    ]).sort_values('date').tail(window)
    points = 0
    for _, m in all_matches.iterrows():
        if m['home_team'] == team:
            if m['home_score'] > m['away_score']:
                points += 3
            elif m['home_score'] == m['away_score']:
                points += 1
        else:
            if m['away_score'] > m['home_score']:
                points += 3
            elif m['away_score'] == m['home_score']:
                points += 1
    return points / window if len(all_matches) > 0 else 1.5

def filter_worldcup_and_qualifiers(df, start_year=1990, end_year=2025):
    mask = df['tournament'].str.contains('World Cup', case=False, na=False)
    wc_data = df[mask].copy()
    wc_data = wc_data[(wc_data['date'].dt.year >= start_year) & (wc_data['date'].dt.year <= end_year)]
    return wc_data

def build_training_data(df, elo_df, start_year=1990, end_year=2025):
    """Construye X, y usando valor de mercado histórico si existe, sino el de 2026."""
    # Intentar cargar histórico
    market_df = load_historical_market_values()
    use_historical = market_df is not None
    if not use_historical:
        market_dict = load_market_values()  # fallback a 2026
    
    df_filtered = filter_worldcup_and_qualifiers(df, start_year, end_year)
    features = []
    home_goals = []
    away_goals = []
    
    for idx, row in df_filtered.iterrows():
        match_date = row['date']
        home = row['home_team']
        away = row['away_team']
        neutral = 1 if 'neutral' in row and row['neutral'] == 1 else 0
        
        elo_home = get_last_elo_before_date(home, match_date, elo_df)
        elo_away = get_last_elo_before_date(away, match_date, elo_df)
        diff_elo = elo_home - elo_away
        
        home_attack = get_team_avg_goals(home, match_date, df, as_home=True)
        away_defense = get_team_avg_goals(away, match_date, df, as_home=False)
        away_attack = get_team_avg_goals(away, match_date, df, as_home=True)
        home_defense = get_team_avg_goals(home, match_date, df, as_home=False)
        form_home = get_team_form(home, match_date, df)
        form_away = get_team_form(away, match_date, df)
        
        if use_historical:
            home_value = get_historical_market_value(home, match_date, market_df)
            away_value = get_historical_market_value(away, match_date, market_df)
        else:
            home_value = get_market_value(home, market_dict)
            away_value = get_market_value(away, market_dict)
        
        features.append([diff_elo, home_attack, away_defense, away_attack, home_defense,
                         form_home, form_away, neutral, home_value, away_value])
        home_goals.append(row['home_score'])
        away_goals.append(row['away_score'])
    
    X = pd.DataFrame(features, columns=['diff_elo', 'home_attack', 'away_defense', 'away_attack',
                                        'home_defense', 'form_home', 'form_away', 'neutral',
                                        'home_value', 'away_value'])
    y_home = np.array(home_goals)
    y_away = np.array(away_goals)
    
    mask = ~(X.isna().any(axis=1) | np.isinf(X).any(axis=1) |
             np.isnan(y_home) | np.isinf(y_home) |
             np.isnan(y_away) | np.isinf(y_away))
    X_clean = X[mask].copy()
    y_home_clean = y_home[mask]
    y_away_clean = y_away[mask]
    
    print(f"Partidos originales: {len(X)}, después de limpiar: {len(X_clean)}")
    return X_clean, y_home_clean, y_away_clean

if __name__ == "__main__":
    df = load_results()
    elo = load_elo_ratings()
    X, y_home, y_away = build_training_data(df, elo)
    print("Ejemplo de características:")
    print(X.head())