import joblib
import statsmodels.api as sm
from data_loader import load_results, load_elo_ratings, build_training_data

def train_poisson_models():
    df = load_results()
    elo_df = load_elo_ratings()
    X, y_home, y_away = build_training_data(df, elo_df)
    
    # Características para modelo de goles local
    X_home = sm.add_constant(X[['diff_elo', 'home_attack', 'away_defense', 'form_home', 'neutral', 'home_value']])
    # Características para modelo de goles visitante
    X_away = sm.add_constant(X[['diff_elo', 'away_attack', 'home_defense', 'form_away', 'neutral', 'away_value']])
    
    model_home = sm.GLM(y_home, X_home, family=sm.families.Poisson()).fit()
    model_away = sm.GLM(y_away, X_away, family=sm.families.Poisson()).fit()
    
    joblib.dump(model_home, 'poisson_home.pkl')
    joblib.dump(model_away, 'poisson_away.pkl')
    print("✅ Modelos Poisson entrenados y guardados (con valor de mercado).")

if __name__ == "__main__":
    train_poisson_models()