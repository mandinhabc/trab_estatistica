from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

def get_modelos():
    """
    Retorna um dicionário com instâncias de modelos de regressão.
    Inclui agora: LinearRegression, RandomForestRegressor, SVR, XGBRegressor,
    DecisionTreeRegressor, KNeighborsRegressor e MLPRegressor.
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
        ),
        'Decision Tree Regressor': DecisionTreeRegressor(
            max_depth=5, random_state=42
        ),
        'KNN Regressor': KNeighborsRegressor(
            n_neighbors=5, n_jobs=-1
        ),
        'MLP Regressor': MLPRegressor(
            hidden_layer_sizes=(100,), activation='relu',
            solver='adam', max_iter=200, random_state=42
        )
    }
