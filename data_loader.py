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

def get_last_elo_before_date(team, date, elo_df, default=1500):
    """Retorna el último Elo del equipo antes de una fecha específica."""
    hist = elo_df[(elo_df['team'] == team) & (elo_df['date'] < date)]
    if not hist.empty:
        return hist.iloc[-1]['elo']
    return default

def get_team_avg_goals(team, date, df, as_home=True, window=5):
    """Promedio de goles anotados (as_home=True) o recibidos (as_home=False) en últimos window partidos antes de date."""
    if as_home:
        matches = df[(df['home_team'] == team) & (df['date'] < date)].sort_values('date').tail(window)
        goals = matches['home_score']
    else:
        matches = df[(df['away_team'] == team) & (df['date'] < date)].sort_values('date').tail(window)
        goals = matches['away_score']
    if not goals.empty:
        return goals.mean()
    return 1.0  # valor neutral

def get_team_form(team, date, df, window=5):
    """Puntos promedio en últimos window partidos (3 victoria, 1 empate, 0 derrota)."""
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
    """Filtra partidos de World Cup y eliminatorias entre dos años."""
    mask = df['tournament'].str.contains('World Cup', case=False, na=False)
    wc_data = df[mask].copy()
    wc_data = wc_data[(wc_data['date'].dt.year >= start_year) & (wc_data['date'].dt.year <= end_year)]
    return wc_data

def build_training_data(df, elo_df, start_year=1990, end_year=2025):
    """Construye X, y para el modelo Poisson."""
    df_filtered = filter_worldcup_and_qualifiers(df, start_year, end_year)
    features = []
    home_goals = []
    away_goals = []
    for idx, row in df_filtered.iterrows():
        match_date = row['date']
        home = row['home_team']
        away = row['away_team']
        neutral = 1 if 'neutral' in row and row['neutral'] == 1 else 0
        # Elo
        elo_home = get_last_elo_before_date(home, match_date, elo_df)
        elo_away = get_last_elo_before_date(away, match_date, elo_df)
        diff_elo = elo_home - elo_away
        # Ataque/defensa
        home_attack = get_team_avg_goals(home, match_date, df, as_home=True)
        away_defense = get_team_avg_goals(away, match_date, df, as_home=False)
        away_attack = get_team_avg_goals(away, match_date, df, as_home=True)
        home_defense = get_team_avg_goals(home, match_date, df, as_home=False)
        form_home = get_team_form(home, match_date, df)
        form_away = get_team_form(away, match_date, df)
        features.append([diff_elo, home_attack, away_defense, away_attack, home_defense, form_home, form_away, neutral])
        home_goals.append(row['home_score'])
        away_goals.append(row['away_score'])
    X = pd.DataFrame(features, columns=['diff_elo', 'home_attack', 'away_defense', 'away_attack', 'home_defense', 'form_home', 'form_away', 'neutral'])
    y_home = np.array(home_goals)
    y_away = np.array(away_goals)
    return X, y_home, y_away

if __name__ == "__main__":
    df = load_results()
    elo = load_elo_ratings()
    X, y_home, y_away = build_training_data(df, elo)
    print(f"Datos de entrenamiento: {X.shape[0]} partidos")
    print(X.head())