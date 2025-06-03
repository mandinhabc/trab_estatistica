# src/main.py

import pandas as pd
import numpy as np
import time

from src.preprocessing import preparar_dados
from src.models import get_modelos
from src.evaluation import avaliar_modelo

# 1. Carregar dados (use caminho relativo à raiz)
df = pd.read_csv(r'C:\Users\abarb\OneDrive\Faculdade\trab_estatistica\dataset\Top_spotify_songs.csv')

# 2. Pré-processar e obter X_train_prepared, X_test_prepared, y_train, y_test
X_train, X_test, y_train, y_test = preparar_dados(df, 'popularity')

# 3. Obter dicionário de modelos de regressão
modelos = get_modelos()

for nome, modelo in modelos.items():
    # 4. Treinar no conjunto de treino
    start = time.time()
    modelo.fit(X_train, y_train)
    tempo_treino = time.time() - start

    # 5. Previsões no treino (necessário para resíduos)
    y_train_pred = modelo.predict(X_train)

    # 6. Previsões no teste
    y_test_pred = modelo.predict(X_test)

    # 7. Avaliar com MPIW (supondo alpha=0.95)
    resultados = avaliar_modelo(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_true=y_test,
        y_pred=y_test_pred,
        alpha=0.95
    )

    # 8. Exibir métricas
    print(f"{nome}:")
    print(f"  • RMSE = {resultados['RMSE']:.4f}")
    print(f"  • R²   = {resultados['R2']:.4f}")
    print(f"  • MPIW = {resultados['MPIW']:.4f} (95 % PI)\n"
          f"  • Tempo de treino = {tempo_treino:.1f}s\n")
