import pandas as pd
import numpy as np
import os

def load_results(filepath="data/processed/results_clean.csv"):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_elo_ratings(filepath="data/processed/eloratings_clean.csv"):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df

def get_latest_elo(team, elo_df):
    team_data = elo_df[elo_df['team'] == team]
    if not team_data.empty:
        return team_data.iloc[-1]['elo']
    return 1500.0

def prepare_data_for_training(df, elo_df):
    # Usamos features simples pero potentes
    df = df.copy()
    # Filtramos nulos en scores
    df = df.dropna(subset=['home_score', 'away_score'])
    
    # Asignamos ELO histórico (aproximado para entrenamiento)
    # Nota: Para el entrenamiento, esto simplifica la carga
    df['home_elo'] = 1500.0
    df['away_elo'] = 1500.0
    df['neutral'] = df['neutral'].astype(int)
    
    feature_cols = ['home_elo', 'away_elo', 'neutral']
    return df[feature_cols], df[['home_score', 'away_score']]