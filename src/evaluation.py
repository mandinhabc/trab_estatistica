# src/evaluation.py

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm

def avaliar_modelo(y_train, y_train_pred, y_true, y_pred, alpha: float = 0.95):
    
    # 1. RMSE e R2 (no teste)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)

    # 2. Estimar sigma a partir dos resíduos no treino
    resid_train = y_train - y_train_pred
    sigma = np.std(resid_train, ddof=1)  # ddof=1 para amostral

    # 3. z-valor para nível de confiança alpha
    z = norm.ppf(1.0 - (1.0 - alpha) / 2.0)

    # 4. Largura do intervalo (constante para cada ponto)
    mpiw = 2.0 * z * sigma

    return {
        'RMSE': rmse,
        'R2': r2,
        'MPIW': mpiw
    }