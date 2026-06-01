import joblib
from sklearn.linear_model import PoissonRegressor
from data_loader import load_results, load_elo_ratings, prepare_data_for_training

def train_model():
    print("Cargando datos...")
    df = load_results()
    elo_df = load_elo_ratings()
    
    X, y = prepare_data_for_training(df, elo_df)
    
    print("Entrenando modelos Poisson...")
    # Modelo para goles locales
    model_home = PoissonRegressor(alpha=1e-5)
    model_home.fit(X, y['home_score'])
    
    # Modelo para goles visitantes
    model_away = PoissonRegressor(alpha=1e-5)
    model_away.fit(X, y['away_score'])
    
    joblib.dump(model_home, 'model_home.pkl')
    joblib.dump(model_away, 'model_away.pkl')
    print("✅ Modelos guardados: model_home.pkl y model_away.pkl")

if __name__ == "__main__":
    train_model()