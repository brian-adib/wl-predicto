import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
import statsmodels.api as sm
from data_loader import load_results, load_elo_ratings, build_training_data

def dixon_coles_likelihood(rho, y_home, y_away, lambda_home, lambda_away):
    """Log-verosimilitud negativa para Dixon-Coles."""
    log_like = 0.0
    n = len(y_home)
    for i in range(n):
        h = y_home[i]
        a = y_away[i]
        lh = lambda_home[i]
        la = lambda_away[i]
        
        prob = poisson.pmf(h, lh) * poisson.pmf(a, la)
        
        if (h == 0 and a == 0) or (h == 0 and a == 1) or (h == 1 and a == 0) or (h == 1 and a == 1):
            correction = 1 - lh * la * rho
            if correction > 0:
                prob *= correction
            else:
                prob = 1e-10
        if prob <= 0:
            prob = 1e-10
        log_like += np.log(prob)
    return -log_like

def calibrate_rho():
    print("Cargando datos...")
    df = load_results()
    elo_df = load_elo_ratings()
    X, y_home, y_away = build_training_data(df, elo_df)
    
    # Asegurar que y_home e y_away sean arrays planos
    if isinstance(y_home, pd.Series):
        y_home = y_home.values
    if isinstance(y_away, pd.Series):
        y_away = y_away.values
    
    print("Ajustando modelos Poisson...")
    X_home = sm.add_constant(X[['diff_elo', 'home_attack', 'away_defense', 'form_home', 'neutral', 'home_value']])
    X_away = sm.add_constant(X[['diff_elo', 'away_attack', 'home_defense', 'form_away', 'neutral', 'away_value']])
    
    model_home = sm.GLM(y_home, X_home, family=sm.families.Poisson()).fit()
    model_away = sm.GLM(y_away, X_away, family=sm.families.Poisson()).fit()
    
    # Predecir lambdas y convertir a arrays numpy
    lambda_home = model_home.predict(X_home)
    lambda_away = model_away.predict(X_away)
    lambda_home = np.maximum(lambda_home.values, 0.1)  # .values convierte a numpy array
    lambda_away = np.maximum(lambda_away.values, 0.1)
    
    print("Optimizando rho...")
    res = minimize_scalar(dixon_coles_likelihood, args=(y_home, y_away, lambda_home, lambda_away),
                          bounds=(-0.5, 0.5), method='bounded')
    rho_opt = res.x
    print(f"✅ Valor óptimo de rho: {rho_opt:.4f}")
    print(f"Log-verosimilitud máxima: {-res.fun:.2f}")
    
    with open('rho_calibrated.txt', 'w') as f:
        f.write(str(rho_opt))
    print("Rho guardado en rho_calibrated.txt")

if __name__ == "__main__":
    calibrate_rho()