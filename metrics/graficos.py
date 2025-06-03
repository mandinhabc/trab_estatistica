import matplotlib.pyplot as plt

# Dados dos modelos
modelos = ['Linear Regression', 'Random Forest', 'XGBoost']
rmse = [12.5927, 5.9281, 7.9614]
r2 = [0.3616, 0.8585, 0.7448]
mpiw = [49.4126, 8.7915, 28.8064]
tempo = [3.2, 701.0, 6.0]

# Gráfico 1: RMSE
plt.figure(figsize=(6, 4))
plt.bar(modelos, rmse, color='skyblue')
plt.title("RMSE por Modelo")
plt.ylabel("RMSE")
plt.ylim(0, max(rmse) + 5)
plt.grid(True, axis='y')
plt.show()

# Gráfico 2: R²
plt.figure(figsize=(6, 4))
plt.bar(modelos, r2, color='lightgreen')
plt.title("R² por Modelo")
plt.ylabel("R²")
plt.ylim(0, 1)
plt.grid(True, axis='y')
plt.show()

# Gráfico 3: MPIW
plt.figure(figsize=(6, 4))
plt.bar(modelos, mpiw, color='salmon')
plt.title("MPIW por Modelo (95%)")
plt.ylabel("MPIW")
plt.ylim(0, max(mpiw) + 10)
plt.grid(True, axis='y')
plt.show()

# Gráfico 4: Tempo de Treinamento
plt.figure(figsize=(6, 4))
plt.bar(modelos, tempo, color='orange')
plt.title("Tempo de Treinamento por Modelo")
plt.ylabel("Tempo (s)")
plt.ylim(0, max(tempo) + 100)
plt.grid(True, axis='y')
plt.show()
