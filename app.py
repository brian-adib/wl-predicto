import streamlit as st
import pandas as pd
import numpy as np
import joblib
from data_loader import load_results, load_elo_ratings
import os

st.set_page_config(page_title="World Cup Predictor", layout="wide")
st.title("⚽ Predicción de Marcadores del Mundial")

@st.cache_resource
def load_model():
    if not os.path.exists("worldcup_model.pkl"):
        st.error("Ejecuta primero python train_model.py")
        return None, None, None
    model = joblib.load("worldcup_model.pkl")
    le = joblib.load("label_encoder.pkl")
    # Obtener la lista de pares (home, away) que el modelo conoce
    max_goals = 15  # solo para decodificar
    class_pairs = []
    for encoded_class in range(len(le.classes_)):
        decoded = le.inverse_transform([encoded_class])[0]
        home_score = decoded // (max_goals + 1)
        away_score = decoded % (max_goals + 1)
        class_pairs.append((int(home_score), int(away_score)))
    return model, le, class_pairs

model, le, class_pairs = load_model()
if model is None:
    st.stop()

@st.cache_data
def load_data():
    return load_results()

@st.cache_data
def load_elo():
    return load_elo_ratings()

df = load_data()
elo_df = load_elo()

years = sorted(df['date'].dt.year.unique())
year = st.selectbox("Selecciona el año del Mundial a predecir:", years)

# Equipos participantes (últimos 12 años antes del torneo)
wc_matches = df[df['tournament'].str.contains('World Cup', na=False)]
recent_wc = wc_matches[wc_matches['date'].dt.year.between(year-12, year-1)]
teams = sorted(set(recent_wc['home_team']).union(set(recent_wc['away_team'])))
st.sidebar.write(f"Equipos participantes: {len(teams)}")

def get_last_elo(team, as_of_year):
    cutoff = pd.Timestamp(f"{as_of_year}-01-01")
    team_elo = elo_df[(elo_df['team'] == team) & (elo_df['date'] < cutoff)]
    if not team_elo.empty:
        return team_elo.iloc[-1]['elo']
    return 1500

# Simulación de fixtures (fase de grupos)
num_groups = 8
teams_per_group = 4
if len(teams) >= num_groups * teams_per_group:
    elo_dict = {t: get_last_elo(t, year) for t in teams}
    selected_teams = sorted(teams, key=lambda t: elo_dict.get(t, 1500), reverse=True)[:num_groups*teams_per_group]
else:
    selected_teams = teams

groups = [selected_teams[i*teams_per_group:(i+1)*teams_per_group] for i in range(num_groups)]
matches = []
for gidx, group in enumerate(groups):
    for j in range(len(group)):
        for k in range(j+1, len(group)):
            matches.append({'fase': f"Grupo {chr(65+gidx)}", 'home': group[j], 'away': group[k]})
            matches.append({'fase': f"Grupo {chr(65+gidx)}", 'home': group[k], 'away': group[j]})

def get_team_stats(team, as_of_year):
    elo_rating = get_last_elo(team, as_of_year)
    cutoff = pd.Timestamp(f"{as_of_year}-01-01")
    home_matches = df[(df['home_team'] == team) & (df['date'] < cutoff)].sort_values('date').tail(5)
    attack = home_matches['home_score'].mean() if not home_matches.empty else 1.0
    away_matches = df[(df['away_team'] == team) & (df['date'] < cutoff)].sort_values('date').tail(5)
    defense = away_matches['away_score'].mean() if not away_matches.empty else 1.0
    all_matches = pd.concat([
        home_matches[['date', 'home_team', 'away_team', 'home_score', 'away_score']],
        away_matches[['date', 'home_team', 'away_team', 'home_score', 'away_score']]
    ]).sort_values('date').tail(5)
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
    form = points / 5 if len(all_matches) > 0 else 1.5
    return elo_rating, attack, defense, form

if st.button("Generar predicciones"):
    results = []
    for match in matches:
        home = match['home']
        away = match['away']
        home_elo, home_attack, _, home_form = get_team_stats(home, year)
        away_elo, _, away_defense, _ = get_team_stats(away, year)
        neutral = 0
        features = np.array([[home_elo, away_elo, home_attack, away_defense, home_form, neutral]])
        proba = model.predict_proba(features)[0]

        # Obtener los 10 índices con mayor probabilidad
        top_indices = np.argsort(proba)[::-1][:10]
        top_scores = []
        for idx in top_indices:
            h, a = class_pairs[idx]
            prob = proba[idx] * 100
            top_scores.append(f"{h}-{a}: {prob:.2f}%")

        # Calcular probabilidades marginales (local gana, empate, visitante gana)
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        for (h, a), p in zip(class_pairs, proba):
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p

        results.append({
            "Fase": match['fase'],
            "Local": home,
            "Visitante": away,
            "Top 10": "; ".join(top_scores),
            "Local gana": f"{home_win*100:.1f}%",
            "Empate": f"{draw*100:.1f}%",
            "Visitante gana": f"{away_win*100:.1f}%"
        })

    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)
    for _, row in df_results.iterrows():
        with st.expander(f"{row['Local']} vs {row['Visitante']} ({row['Fase']})"):
            st.write(row['Top 10'])
            col1, col2, col3 = st.columns(3)
            col1.metric("Gana Local", row['Local gana'])
            col2.metric("Empate", row['Empate'])
            col3.metric("Gana Visitante", row['Visitante gana'])