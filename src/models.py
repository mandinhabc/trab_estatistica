# src/models.py

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def get_modelos_regressao():
    """
    Retorna um dicionário com instâncias de modelos de regressão.
    Inclui agora: LinearRegression, RandomForestRegressor, SVR e XGBRegressor.
    """
    return {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=30, max_depth=8, n_jobs=-1, random_state=42
        ),
        'SVR': SVR(
            kernel='rbf',       # padrão
            C=1.0,              # penalização
            epsilon=0.1,        # margem de tolerância
            cache_size=200,     # em MB
            max_iter=100000     # limitar iterações para não travar
        ),
        'XGBoost Regressor': XGBRegressor(
            n_estimators=50, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbosity=0
        )
    }
