import streamlit as st
import pandas as pd
import numpy as np
import joblib
from data_loader import load_results, load_elo_ratings, get_last_elo_before_date, get_team_avg_goals, get_team_form, load_market_values, get_market_value
from datetime import datetime
from dixon_coles import build_dixon_coles_matrix

# ==================== CONFIGURACIÓN ====================
try:
    with open('rho_calibrated.txt', 'r') as f:
        RHO = float(f.read().strip())
except FileNotFoundError:
    RHO = -0.13

st.set_page_config(page_title="World Cup 2026 Predictor", layout="wide")
st.title("🏆 Mundial 2026 - Predictor de Marcadores con Poisson (Dixon-Coles)")

# ==================== BANDERAS ====================
flag_map = {
    "México": "🇲🇽", "Sudáfrica": "🇿🇦", "Corea del Sur": "🇰🇷", "República Checa": "🇨🇿",
    "Canadá": "🇨🇦", "Bosnia y Herzegovina": "🇧🇦", "Qatar": "🇶🇦", "Suiza": "🇨🇭",
    "Brasil": "🇧🇷", "Marruecos": "🇲🇦", "Haití": "🇭🇹", "Escocia": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Estados Unidos": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Turquía": "🇹🇷",
    "Alemania": "🇩🇪", "Curazao": "🇨🇼", "Costa de Marfil": "🇨🇮", "Ecuador": "🇪🇨",
    "Países Bajos": "🇳🇱", "Japón": "🇯🇵", "Suecia": "🇸🇪", "Túnez": "🇹🇳",
    "Bélgica": "🇧🇪", "Egipto": "🇪🇬", "Irán": "🇮🇷", "Nueva Zelanda": "🇳🇿",
    "España": "🇪🇸", "Cabo Verde": "🇨🇻", "Arabia Saudita": "🇸🇦", "Uruguay": "🇺🇾",
    "Francia": "🇫🇷", "Senegal": "🇸🇳", "Irak": "🇮🇶", "Noruega": "🇳🇴",
    "Argentina": "🇦🇷", "Argelia": "🇩🇿", "Austria": "🇦🇹", "Jordania": "🇯🇴",
    "Portugal": "🇵🇹", "República Democrática del Congo": "🇨🇩", "Uzbekistán": "🇺🇿", "Colombia": "🇨🇴",
    "Inglaterra": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croacia": "🇭🇷", "Ghana": "🇬🇭", "Panamá": "🇵🇦"
}

def flag(name):
    return flag_map.get(name, "🏁") + " " + name

# ==================== CARGAR MODELOS Y DATOS ====================
@st.cache_resource
def load_models():
    home_model = joblib.load('poisson_home.pkl')
    away_model = joblib.load('poisson_away.pkl')
    return home_model, away_model

home_model, away_model = load_models()

@st.cache_data
def load_all_data():
    return load_results(), load_elo_ratings(), load_market_values()

df, elo_df, market_dict = load_all_data()
tournament_start = pd.Timestamp("2026-06-11")

# ==================== GRUPOS ====================
GROUPS = {
    "A": ["México", "Sudáfrica", "Corea del Sur", "República Checa"],
    "B": ["Canadá", "Bosnia y Herzegovina", "Qatar", "Suiza"],
    "C": ["Brasil", "Marruecos", "Haití", "Escocia"],
    "D": ["Estados Unidos", "Paraguay", "Australia", "Turquía"],
    "E": ["Alemania", "Curazao", "Costa de Marfil", "Ecuador"],
    "F": ["Países Bajos", "Japón", "Suecia", "Túnez"],
    "G": ["Bélgica", "Egipto", "Irán", "Nueva Zelanda"],
    "H": ["España", "Cabo Verde", "Arabia Saudita", "Uruguay"],
    "I": ["Francia", "Senegal", "Irak", "Noruega"],
    "J": ["Argentina", "Argelia", "Austria", "Jordania"],
    "K": ["Portugal", "República Democrática del Congo", "Uzbekistán", "Colombia"],
    "L": ["Inglaterra", "Croacia", "Ghana", "Panamá"]
}

# Mapeo de nombres a inglés para búsqueda interna
NAME_MAPPING = {
    "México": "Mexico", "Sudáfrica": "South Africa", "Corea del Sur": "South Korea", "República Checa": "Czech Republic",
    "Canadá": "Canada", "Bosnia y Herzegovina": "Bosnia and Herzegovina", "Qatar": "Qatar", "Suiza": "Switzerland",
    "Brasil": "Brazil", "Marruecos": "Morocco", "Haití": "Haiti", "Escocia": "Scotland",
    "Estados Unidos": "United States", "Paraguay": "Paraguay", "Australia": "Australia", "Turquía": "Turkey",
    "Alemania": "Germany", "Curazao": "Curaçao", "Costa de Marfil": "Ivory Coast", "Ecuador": "Ecuador",
    "Países Bajos": "Netherlands", "Japón": "Japan", "Suecia": "Sweden", "Túnez": "Tunisia",
    "Bélgica": "Belgium", "Egipto": "Egypt", "Irán": "Iran", "Nueva Zelanda": "New Zealand",
    "España": "Spain", "Cabo Verde": "Cape Verde", "Arabia Saudita": "Saudi Arabia", "Uruguay": "Uruguay",
    "Francia": "France", "Senegal": "Senegal", "Irak": "Iraq", "Noruega": "Norway",
    "Argentina": "Argentina", "Argelia": "Algeria", "Austria": "Austria", "Jordania": "Jordan",
    "Portugal": "Portugal", "República Democrática del Congo": "Democratic Republic of the Congo", "Uzbekistán": "Uzbekistan", "Colombia": "Colombia",
    "Inglaterra": "England", "Croacia": "Croatia", "Ghana": "Ghana", "Panamá": "Panama"
}

def map_team(name):
    return NAME_MAPPING.get(name, name)

def get_team_features(team, date):
    team_eng = map_team(team)
    elo = get_last_elo_before_date(team_eng, date, elo_df)
    home_attack = get_team_avg_goals(team_eng, date, df, as_home=True)
    away_defense = get_team_avg_goals(team_eng, date, df, as_home=False)
    away_attack = get_team_avg_goals(team_eng, date, df, as_home=True)
    home_defense = get_team_avg_goals(team_eng, date, df, as_home=False)
    form_home = get_team_form(team_eng, date, df)
    form_away = get_team_form(team_eng, date, df)
    value = get_market_value(team, market_dict)
    return elo, home_attack, away_defense, away_attack, home_defense, form_home, form_away, value

def predict_match(home_team, away_team, match_date, neutral=1):
    h_elo, h_att, h_def_away, h_att_away, h_def_home, h_form_home, _, h_value = get_team_features(home_team, match_date)
    a_elo, a_att, a_def_away, a_att_away, a_def_home, _, a_form_away, a_value = get_team_features(away_team, match_date)
    diff_elo = h_elo - a_elo
    X_home = pd.DataFrame([[1, diff_elo, h_att, a_def_away, h_form_home, neutral, h_value]],
                          columns=['const', 'diff_elo', 'home_attack', 'away_defense', 'form_home', 'neutral', 'home_value'])
    X_away = pd.DataFrame([[1, diff_elo, a_att_away, h_def_home, a_form_away, neutral, a_value]],
                          columns=['const', 'diff_elo', 'away_attack', 'home_defense', 'form_away', 'neutral', 'away_value'])
    lambda_home = max(home_model.predict(X_home)[0], 0.1)
    lambda_away = max(away_model.predict(X_away)[0], 0.1)
    max_goals = 7
    prob_matrix = build_dixon_coles_matrix(lambda_home, lambda_away, max_goals=max_goals, rho=RHO)
    flat = prob_matrix.flatten()
    top_idx = np.argsort(flat)[::-1][:10]
    top_scores = [f"{i//(max_goals+1)}-{i%(max_goals+1)}: {flat[i]*100:.2f}%" for i in top_idx]
    home_win = np.sum(prob_matrix[np.tril_indices_from(prob_matrix, k=-1)])
    away_win = np.sum(prob_matrix[np.triu_indices_from(prob_matrix, k=1)])
    draw = np.sum(np.diag(prob_matrix))
    return top_scores, home_win, draw, away_win

# ==================== FUNCIÓN PRINCIPAL DE SIMULACIÓN ====================
def run_simulation():
    # Fase de grupos
    team_points = {team: 0 for group in GROUPS.values() for team in group}
    team_gd = {team: 0 for team in team_points}
    group_matches_detail = {}

    for group_name, teams in GROUPS.items():
        matches = []
        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                home, away = teams[i], teams[j]
                top_scores, hw, dr, aw = predict_match(home, away, tournament_start, neutral=1)
                # Asignar puntos para clasificación
                if hw > aw and hw > dr:
                    team_points[home] += 3
                    team_gd[home] += 1
                    team_gd[away] -= 1
                elif aw > hw and aw > dr:
                    team_points[away] += 3
                    team_gd[away] += 1
                    team_gd[home] -= 1
                else:
                    team_points[home] += 1
                    team_points[away] += 1
                matches.append((home, away, top_scores, hw, dr, aw))
        group_matches_detail[group_name] = matches

    # Clasificación
    firsts, seconds, thirds = [], [], []
    for group_name, teams in GROUPS.items():
        sorted_teams = sorted(teams, key=lambda t: (team_points[t], team_gd[t]), reverse=True)
        firsts.append(sorted_teams[0])
        seconds.append(sorted_teams[1])
        thirds.append(sorted_teams[2])
    thirds_sorted = sorted(thirds, key=lambda t: (team_points[t], team_gd[t]), reverse=True)
    best_thirds = thirds_sorted[:8]
    qualified = firsts + seconds + best_thirds
    # Orden por Elo para eliminatorias
    team_elo = {}
    for team in qualified:
        team_eng = map_team(team)
        team_elo[team] = get_last_elo_before_date(team_eng, tournament_start, elo_df)
    qualified_sorted = sorted(qualified, key=lambda t: team_elo.get(t, 1500), reverse=True)
    # Ronda 1 (16vos de final) – 32 equipos → 16 partidos
    matches_round1 = [(qualified_sorted[i], qualified_sorted[31-i]) for i in range(16)]
    # Procesar rondas eliminatorias
    def process_round(matches, days_offset):
        winners = []
        round_matches = []
        for home, away in matches:
            top_scores, hw, dr, aw = predict_match(home, away, tournament_start + pd.Timedelta(days=days_offset), neutral=1)
            if hw > aw and hw > dr:
                winner = home
            elif aw > hw and aw > dr:
                winner = away
            else:
                winner = home
            winners.append(winner)
            round_matches.append((home, away, top_scores, hw, dr, aw, winner))
        return winners, round_matches

    w1, r1 = process_round(matches_round1, 15)
    # Octavos
    matches_round2 = [(w1[i], w1[i+1]) for i in range(0, len(w1), 2)]
    w2, r2 = process_round(matches_round2, 20)
    # Cuartos
    matches_round3 = [(w2[i], w2[i+1]) for i in range(0, len(w2), 2)]
    w3, r3 = process_round(matches_round3, 25)
    # Semifinales
    matches_round4 = [(w3[i], w3[i+1]) for i in range(0, len(w3), 2)]
    w4, r4 = process_round(matches_round4, 28)
    # Final
    home_final, away_final = w4[0], w4[1]
    top_scores_final, hw_f, dr_f, aw_f = predict_match(home_final, away_final, tournament_start + pd.Timedelta(days=32), neutral=1)
    if hw_f > aw_f and hw_f > dr_f:
        champion = home_final
    elif aw_f > hw_f and aw_f > dr_f:
        champion = away_final
    else:
        champion = home_final

    return {
        "group_matches": group_matches_detail,
        "group_points": team_points,
        "group_gd": team_gd,
        "qualified": qualified,
        "round1": r1,
        "round2": r2,
        "round3": r3,
        "round4": r4,
        "final": (home_final, away_final, top_scores_final, hw_f, dr_f, aw_f, champion)
    }

# ==================== INTERFAZ ====================
st.header("📋 Grupos del Mundial 2026")
# Mostrar grupos en dos columnas
group_names = list(GROUPS.keys())
mid = len(group_names) // 2
col1, col2 = st.columns(2)
with col1:
    for g in group_names[:mid]:
        teams = GROUPS[g]
        st.markdown(f"**Grupo {g}:** " + " | ".join([flag(t) for t in teams]))
with col2:
    for g in group_names[mid:]:
        teams = GROUPS[g]
        st.markdown(f"**Grupo {g}:** " + " | ".join([flag(t) for t in teams]))

st.markdown("---")

# Botón para iniciar simulación (solo una vez)
if st.button("🔮 Generar predicciones para todo el torneo"):
    with st.spinner("Simulando el torneo (esto puede tomar unos segundos)..."):
        results = run_simulation()
        st.session_state['simulation'] = results
        st.session_state['simulation_done'] = True

if st.session_state.get('simulation_done', False):
    results = st.session_state['simulation']

    # ========== FASE DE GRUPOS ==========
    st.subheader("📊 Fase de grupos")
    # Mostrar grupos en dos columnas
    group_list = list(results["group_matches"].items())
    mid = len(group_list) // 2
    col_left, col_right = st.columns(2)
    with col_left:
        for group_name, matches in group_list[:mid]:
            st.markdown(f"#### Grupo {group_name}")
            for home, away, top_scores, hw, dr, aw in matches:
                st.markdown(f"**{flag(home)} vs {flag(away)}**")
                st.markdown("**Top 10 marcadores:**")
                for ts in top_scores:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;{ts}")
                st.markdown(f"🏠 Local: {hw*100:.1f}% | 🤝 Empate: {dr*100:.1f}% | ✈️ Visitante: {aw*100:.1f}%")
                st.markdown("---")
    with col_right:
        for group_name, matches in group_list[mid:]:
            st.markdown(f"#### Grupo {group_name}")
            for home, away, top_scores, hw, dr, aw in matches:
                st.markdown(f"**{flag(home)} vs {flag(away)}**")
                st.markdown("**Top 10 marcadores:**")
                for ts in top_scores:
                    st.markdown(f"&nbsp;&nbsp;&nbsp;{ts}")
                st.markdown(f"🏠 Local: {hw*100:.1f}% | 🤝 Empate: {dr*100:.1f}% | ✈️ Visitante: {aw*100:.1f}%")
                st.markdown("---")

    # ========== TABLA DE POSICIONES ==========
    st.subheader("📈 Clasificación de grupos")
    pos_data = []
    for group_name, teams in GROUPS.items():
        for team in teams:
            pos_data.append({
                "Grupo": group_name,
                "Equipo": flag(team),
                "Puntos": results["group_points"][team],
                "DG": results["group_gd"][team]
            })
    df_pos = pd.DataFrame(pos_data)
    # Mostrar ordenado por grupo y puntos
    df_pos = df_pos.sort_values(["Grupo", "Puntos", "DG"], ascending=[True, False, False])
    st.dataframe(df_pos, use_container_width=True, hide_index=True)

    # ========== ELIMINATORIAS ==========
    st.subheader("🏆 Fases eliminatorias")
    # 16vos de final
    st.markdown("### 16vos de final")
    for idx, (home, away, top_scores, hw, dr, aw, winner) in enumerate(results["round1"], 1):
        st.markdown(f"**Partido {idx}:** {flag(home)} vs {flag(away)}")
        st.markdown("**Top 10 marcadores:**")
        for ts in top_scores:
            st.markdown(f"&nbsp;&nbsp;&nbsp;{ts}")
        st.markdown(f"🏠 Local: {hw*100:.1f}% | 🤝 Empate: {dr*100:.1f}% | ✈️ Visitante: {aw*100:.1f}%")
        st.markdown(f"🏅 **Ganador pronosticado:** {flag(winner)}")
        st.markdown("---")

    # Octavos
    st.markdown("### Octavos de final")
    for idx, (home, away, top_scores, hw, dr, aw, winner) in enumerate(results["round2"], 1):
        st.markdown(f"**Partido {idx}:** {flag(home)} vs {flag(away)}")
        st.markdown("**Top 10 marcadores:**")
        for ts in top_scores:
            st.markdown(f"&nbsp;&nbsp;&nbsp;{ts}")
        st.markdown(f"🏠 Local: {hw*100:.1f}% | 🤝 Empate: {dr*100:.1f}% | ✈️ Visitante: {aw*100:.1f}%")
        st.markdown(f"🏅 **Ganador pronosticado:** {flag(winner)}")
        st.markdown("---")

    # Cuartos
    st.markdown("### Cuartos de final")
    for idx, (home, away, top_scores, hw, dr, aw, winner) in enumerate(results["round3"], 1):
        st.markdown(f"**Partido {idx}:** {flag(home)} vs {flag(away)}")
        st.markdown("**Top 10 marcadores:**")
        for ts in top_scores:
            st.markdown(f"&nbsp;&nbsp;&nbsp;{ts}")
        st.markdown(f"🏠 Local: {hw*100:.1f}% | 🤝 Empate: {dr*100:.1f}% | ✈️ Visitante: {aw*100:.1f}%")
        st.markdown(f"🏅 **Ganador pronosticado:** {flag(winner)}")
        st.markdown("---")

    # Semifinales
    st.markdown("### Semifinales")
    for idx, (home, away, top_scores, hw, dr, aw, winner) in enumerate(results["round4"], 1):
        st.markdown(f"**Partido {idx}:** {flag(home)} vs {flag(away)}")
        st.markdown("**Top 10 marcadores:**")
        for ts in top_scores:
            st.markdown(f"&nbsp;&nbsp;&nbsp;{ts}")
        st.markdown(f"🏠 Local: {hw*100:.1f}% | 🤝 Empate: {dr*100:.1f}% | ✈️ Visitante: {aw*100:.1f}%")
        st.markdown(f"🏅 **Ganador pronosticado:** {flag(winner)}")
        st.markdown("---")

    # Final
    st.markdown("### Final")
    home, away, top_scores, hw, dr, aw, champion = results["final"]
    st.markdown(f"**{flag(home)} vs {flag(away)}**")
    st.markdown("**Top 10 marcadores:**")
    for ts in top_scores:
        st.markdown(f"&nbsp;&nbsp;&nbsp;{ts}")
    st.markdown(f"🏠 Local: {hw*100:.1f}% | 🤝 Empate: {dr*100:.1f}% | ✈️ Visitante: {aw*100:.1f}%")
    st.success(f"🏆 **El campeón pronosticado es: {flag(champion)}** 🏆")

else:
    st.info("Presiona el botón 'Generar predicciones' para comenzar la simulación del torneo.")