import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_data(df):
    # Separa colunas numéricas e categóricas
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

    # Normaliza colunas numéricas
    scaler = MinMaxScaler()
    normalized_numeric = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]),
        columns=numeric_cols
    )

    # Arredonda as colunas numéricas para 6 casas decimais
    normalized_numeric = normalized_numeric.round(6)

    # One-hot encode categóricas e converte para inteiro
    encoded_categorical = pd.get_dummies(df[categorical_cols], drop_first=False)
    encoded_categorical = encoded_categorical.astype(int)  # <- força int, não float

    # Junta os dados
    final_df = pd.concat([normalized_numeric, encoded_categorical], axis=1)

    return final_df
