# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preparar_dados(df: pd.DataFrame, target_col: str):
    # Converter datas (caso string)
    if df['snapshot_date'].dtype == object:
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    if df['album_release_date'].dtype == object:
        df['album_release_date'] = pd.to_datetime(df['album_release_date'])

    # Definir X e y (remover colunas irrelevantes)
    X = df.drop(columns=[
        'spotify_id', 'name', 'artists', 'album_name',
        'snapshot_date', 'album_release_date', target_col
    ])
    y = df[target_col]

    # Colunas numéricas e categóricas
    num_cols = [
        'daily_rank', 'daily_movement', 'weekly_movement',
        'duration_ms', 'danceability', 'energy', 'key',
        'loudness', 'mode', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
    ]
    cat_cols = ['country', 'is_explicit']

    # Transformers
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ], remainder='drop')

    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Ajustar e transformar
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared  = preprocessor.transform(X_test)

    return X_train_prepared, X_test_prepared, y_train, y_test
