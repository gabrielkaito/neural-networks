import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_data(df, scaler=None):
    # Assume que a última coluna é a saída
    entrada_df = df.iloc[:, :-1]
    saida_df = df.iloc[:, -1:]

    # Separa colunas numéricas e categóricas nas ENTRADAS
    numeric_cols = entrada_df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = entrada_df.select_dtypes(include=["object", "category", "bool"]).columns

    # Normaliza colunas numéricas
    if scaler is None:
        scaler = MinMaxScaler()
        normalized_numeric = pd.DataFrame(
            scaler.fit_transform(entrada_df[numeric_cols]),
            columns=numeric_cols
        ).round(6)
    else:
        normalized_numeric = pd.DataFrame(
            scaler.transform(entrada_df[numeric_cols]),
            columns=numeric_cols
        ).round(6)

    # Codifica colunas categóricas com prefixo "entrada" (se houver)
    if len(categorical_cols) > 0:
        encoded_categorical = pd.get_dummies(
            entrada_df[categorical_cols],
            drop_first=False,
            prefix=["entrada"] * len(categorical_cols)
        ).astype(int)
        entrada_final = pd.concat([normalized_numeric, encoded_categorical], axis=1)
    else:
        entrada_final = normalized_numeric.copy()

    # Codifica saída com prefixo "classe" (se necessário)
    if saida_df.dtypes[0] in ["object", "category", "bool"]:
        saida_final = pd.get_dummies(
            saida_df.iloc[:, 0],
            drop_first=False,
            prefix="classe"
        ).astype(int)
    else:
        saida_final = saida_df.copy()
        if saida_final.shape[1] == 1:
            saida_final.columns = ["saida"]

    # Concatena tudo
    final_df = pd.concat([entrada_final, saida_final], axis=1)

    qtdEntradas = entrada_final.shape[1]
    qtdSaidas = saida_final.shape[1]

    return final_df, qtdEntradas, qtdSaidas, scaler
