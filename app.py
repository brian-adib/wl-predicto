import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson
from data_loader import load_elo_ratings, get_latest_elo

st.set_page_config(page_title="World Cup 2026 Simulator", layout="wide")
st.title("⚽ Simulador Mundial 2026 (Modelo Poisson)")

# Cargar modelos
model_home = joblib.load("model_home.pkl")
model_away = joblib.load("model_away.pkl")
elo_df = load_elo_ratings()

grupos = {
    "Grupo A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "Grupo B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "Grupo C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "Grupo D": ["United States", "Paraguay", "Australia", "Turkey"],
    "Grupo E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "Grupo F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "Grupo G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "Grupo H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "Grupo I": ["France", "Senegal", "Iraq", "Norway"],
    "Grupo J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "Grupo K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "Grupo L": ["England", "Croatia", "Ghana", "Panama"]
}

def predict_score(home, away):
    h_elo = get_latest_elo(home, elo_df)
    a_elo = get_latest_elo(away, elo_df)
    features = pd.DataFrame([[h_elo, a_elo, 1]], columns=['home_elo', 'away_elo', 'neutral'])
    
    lam_home = model_home.predict(features)[0]
    lam_away = model_away.predict(features)[0]
    
    # Calcular prob de victoria
    prob_win = 0
    prob_draw = 0
    prob_loss = 0
    
    # Suma de probabilidades Poisson hasta 6 goles
    for i in range(6):
        for j in range(6):
            p = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away)
            if i > j: prob_win += p
            elif i == j: prob_draw += p
            else: prob_loss += p
            
    return lam_home, lam_away, prob_win, prob_draw, prob_loss

if st.button("🚀 Simular Fase de Grupos"):
    results = []
    for g_name, teams in grupos.items():
        st.subheader(g_name)
        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                h_team, a_team = teams[i], teams[j]
                h_goal, a_goal, pw, pd, pl = predict_score(h_team, a_team)
                results.append({
                    "Grupo": g_name,
                    "Partido": f"{h_team} vs {a_team}",
                    "Marcador Probable": f"{round(h_goal)}-{round(a_goal)}",
                    "Gana Local": f"{pw:.1%}",
                    "Empate": f"{pd:.1%}",
                    "Gana Visita": f"{pl:.1%}"
                })
    
    st.table(pd.DataFrame(results))