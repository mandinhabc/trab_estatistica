# src/evaluation.py

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def avaliar_modelo(y_true, y_pred):
    """
    Retorna um dicionário com RMSE e R2.
    """
    mse = mean_squared_error(y_true, y_pred)  # MSE puro
    rmse = np.sqrt(mse)                      # converte em RMSE
    r2   = r2_score(y_true, y_pred)          # R²

    return {
        'RMSE': rmse,
        'R2': r2
    }
