import pandas as pd
import os

INPUT_DIR = "data/raw/external"
OUTPUT_FILE = "data/processed/historical_team_market_values.csv"

# 1. Cargar national_teams para mapear ID -> nombre del equipo
national_teams = pd.read_csv(os.path.join(INPUT_DIR, "national_teams.csv"))
# Asegurar que tenemos las columnas correctas
if 'national_team_id' not in national_teams.columns or 'name' not in national_teams.columns:
    raise ValueError("national_teams.csv debe tener columnas 'national_team_id' y 'name'")
team_id_to_name = dict(zip(national_teams['national_team_id'], national_teams['name']))

# 2. Cargar players y quedarnos con player_id y current_national_team_id
players = pd.read_csv(os.path.join(INPUT_DIR, "players.csv"))
if 'player_id' not in players.columns or 'current_national_team_id' not in players.columns:
    raise ValueError("players.csv debe tener columnas 'player_id' y 'current_national_team_id'")
players_subset = players[['player_id', 'current_national_team_id']].dropna()

# 3. Cargar player_valuations
valuations = pd.read_csv(os.path.join(INPUT_DIR, "player_valuations.csv"))
valuations['date'] = pd.to_datetime(valuations['date'])

# 4. Unir valuaciones con players para obtener national_team_id
merged = valuations.merge(players_subset, on='player_id', how='inner')
merged = merged.dropna(subset=['current_national_team_id'])
merged['national_team_id'] = merged['current_national_team_id'].astype(int)

# 5. Agregar nombre del equipo
merged['team'] = merged['national_team_id'].map(team_id_to_name)
merged = merged.dropna(subset=['team'])

# 6. Agrupar por equipo y mes (sumando market_value_in_eur)
merged['year_month'] = merged['date'].dt.to_period('M')
monthly = merged.groupby(['team', 'year_month'])['market_value_in_eur'].sum().reset_index()
monthly['date'] = monthly['year_month'].dt.start_time
monthly = monthly[['team', 'date', 'market_value_in_eur']].sort_values(['team', 'date'])

# 7. Guardar
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
monthly.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Archivo generado: {OUTPUT_FILE}")
print(monthly.head())