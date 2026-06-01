import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from data_loader import load_results, load_elo_ratings, prepare_data_for_training

def train_model(exclude_year=2026, max_goals=7):
    df = load_results()
    elo_df = load_elo_ratings()
    X, y, _ = prepare_data_for_training(df, elo_df, exclude_year)

    n_classes = (max_goals + 1) * (max_goals + 1)
    y_combined = y['home_score'] * (max_goals + 1) + y['away_score']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_combined)

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

    joblib.dump(model, 'worldcup_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("✅ Modelo entrenado y guardado.")

if __name__ == "__main__":
    train_model()