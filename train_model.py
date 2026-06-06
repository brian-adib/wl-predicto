import joblib
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data_loader import load_results, load_elo_ratings, build_training_data

def train_xgboost_model(max_goals=7):
    df = load_results()
    elo_df = load_elo_ratings()
    X, y_home, y_away = build_training_data(df, elo_df)
    
    # Combinar goles local y visitante en una única clase (0..(max_goals+1)^2 -1)
    y_combined = y_home * (max_goals + 1) + y_away
    # Recortar valores mayores a max_goals (por si hubiera algún marcador con más de 7 goles)
    y_combined = np.clip(y_combined, 0, (max_goals+1)*(max_goals+1)-1)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_combined)
    
    n_classes = (max_goals + 1) * (max_goals + 1)
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y_encoded)
    
    joblib.dump(model, 'worldcup_model_xgb.pkl')
    joblib.dump(le, 'label_encoder_xgb.pkl')
    print("✅ Modelo XGBoost entrenado y guardado (con valor de mercado).")

if __name__ == "__main__":
    train_xgboost_model()