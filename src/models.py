# src/models.py

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def get_modelos():
    """
    Retorna um dicionário com instâncias de modelos de regressão sem restrições de recursos.
    """
    return {
        'Linear Regression': LinearRegression(),

        'Random Forest Regressor': RandomForestRegressor(
            # Nenhum max_depth ou n_estimators reduzido
            random_state=42,
            n_jobs=-1
        ),

        'SVR': SVR(
            kernel='rbf',   # padrão
            C=1.0,          # pode ajustar via validação cruzada
            epsilon=0.1     # pode ajustar também
            # sem max_iter → vai rodar até convergir
        ),

        'XGBoost Regressor': XGBRegressor(
            # Sem n_estimators ou max_depth limitados
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    }
