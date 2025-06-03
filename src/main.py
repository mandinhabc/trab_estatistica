# src/main.py

import pandas as pd
from src.preprocessing import preparar_dados
from src.models import get_modelos
from src.evaluation import avaliar_modelo

# Use caminho relativo ao chamar python -m src.main na raiz do projeto:
df = pd.read_csv(r'C:\Users\abarb\OneDrive\Faculdade\trab_estatistica\dataset\Top_spotify_songs.csv')

# Obter X_train_prepared, X_test_prepared, y_train e y_test
X_train, X_test, y_train, y_test = preparar_dados(df, 'popularity')

# Dicionário de modelos (definido em src/models.py)
modelos = get_modelos()
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    resultados = avaliar_modelo(y_test, y_pred)
    print(f"{nome}: {resultados}")
