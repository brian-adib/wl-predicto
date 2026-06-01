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
    # Convertir fechas (importante)
    elo_df['date'] = pd.to_datetime(elo_df['date'], errors='coerce')
    # Eliminar filas con fechas inválidas
    elo_df = elo_df.dropna(subset=['date'])
    elo_df = elo_df.sort_values('date')
    # Asegurar columna 'elo'
    if 'rating' in elo_df.columns:
        elo_df = elo_df.rename(columns={'rating': 'elo'})
    # Asegurar que 'elo' sea numérico
    elo_df['elo'] = pd.to_numeric(elo_df['elo'], errors='coerce')
    return elo_df

def merge_elo_to_matches(df, elo_df, initial_elo=1500):
    df = df.copy()
    df['home_elo'] = initial_elo
    df['away_elo'] = initial_elo
    for idx, row in df.iterrows():
        match_date = row['date']
        home = row['home_team']
        away = row['away_team']
        home_hist = elo_df[(elo_df['team'] == home) & (elo_df['date'] < match_date)]
        if not home_hist.empty:
            df.at[idx, 'home_elo'] = home_hist.iloc[-1]['elo']
        away_hist = elo_df[(elo_df['team'] == away) & (elo_df['date'] < match_date)]
        if not away_hist.empty:
            df.at[idx, 'away_elo'] = away_hist.iloc[-1]['elo']
    return df

def compute_rolling_features(df, window=5):
    df['home_attack'] = np.nan
    df['away_defense'] = np.nan
    df['home_form'] = np.nan
    teams = set(df['home_team']).union(set(df['away_team']))
    for team in teams:
        team_home = df[df['home_team'] == team].sort_values('date')
        if not team_home.empty:
            team_home['goals'] = team_home['home_score']
            team_home['rolling'] = team_home['goals'].rolling(window, min_periods=1).mean()
            for idx, val in team_home['rolling'].items():
                df.at[idx, 'home_attack'] = val
        team_away = df[df['away_team'] == team].sort_values('date')
        if not team_away.empty:
            team_away['conceded'] = team_away['home_score']
            team_away['rolling_def'] = team_away['conceded'].rolling(window, min_periods=1).mean()
            for idx, val in team_away['rolling_def'].items():
                df.at[idx, 'away_defense'] = val
        all_matches = pd.concat([
            team_home[['date', 'home_team', 'away_team', 'home_score', 'away_score']],
            team_away[['date', 'home_team', 'away_team', 'home_score', 'away_score']]
        ]).sort_values('date').drop_duplicates(subset='date')
        points = []
        for _, m in all_matches.iterrows():
            if m['home_team'] == team:
                if m['home_score'] > m['away_score']:
                    p = 3
                elif m['home_score'] == m['away_score']:
                    p = 1
                else:
                    p = 0
            else:
                if m['away_score'] > m['home_score']:
                    p = 3
                elif m['away_score'] == m['home_score']:
                    p = 1
                else:
                    p = 0
            points.append(p)
        all_matches['points'] = points
        all_matches['form'] = all_matches['points'].rolling(window, min_periods=1).mean()
        for _, m in all_matches.iterrows():
            idx = m.name
            if idx in df.index:
                if df.loc[idx, 'home_team'] == team or df.loc[idx, 'away_team'] == team:
                    df.at[idx, 'home_form'] = m['form']
    df['home_attack'] = df['home_attack'].fillna(df['home_score'].mean())
    df['away_defense'] = df['away_defense'].fillna(df['away_score'].mean())
    df['home_form'] = df['home_form'].fillna(1.5)
    return df

def filter_worldcup_matches(df, exclude_year=None):
    mask = df['tournament'].str.contains('World Cup', case=False, na=False)
    wc_data = df[mask].copy()
    if exclude_year:
        wc_data = wc_data[wc_data['date'].dt.year < exclude_year]
    return wc_data

def prepare_data_for_training(df, elo_df, exclude_year=2026):
    df_filtered = filter_worldcup_matches(df, exclude_year)
    df_filtered = merge_elo_to_matches(df_filtered, elo_df)
    df_filtered = compute_rolling_features(df_filtered)
    if 'neutral' not in df_filtered.columns:
        df_filtered['neutral'] = 0
    feature_cols = ['home_elo', 'away_elo', 'home_attack', 'away_defense', 'home_form', 'neutral']
    X = df_filtered[feature_cols].fillna(0)
    y = df_filtered[['home_score', 'away_score']]
    return X, y, df_filtered

if __name__ == "__main__":
    df = load_results()
    elo = load_elo_ratings()
    print(f"Partidos totales: {len(df)}")
    X, y, _ = prepare_data_for_training(df, elo, exclude_year=2026)
    print(f"Características: {X.shape}, Objetivos: {y.shape}")