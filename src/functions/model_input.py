from typing import Dict, Any

import polars as pl
from imblearn.over_sampling import SMOTE
import logging

from sklearn.model_selection import train_test_split

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 1. Escalado de variables numéricos
def min_max_scaler_pl(df: pl.DataFrame) -> pl.DataFrame:
    """
    Estandariza las columnas numéricas (excluyendo binarias) de un DataFrame utilizando el método Min-Max Scaler.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame de Polars que se estandarizará.

    Returns
    -------
    pl.DataFrame
        DataFrame estandarizado.
    """
    logger.info("Iniciando la estandarización con Min-Max Scaler...")

    # Identificar las columnas numéricas
    numeric_cols = [name for name, dtype in df.schema.items() if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)]

    # Filtrar solo las columnas numéricas no binarias (excluyendo aquellas que solo toman valores 0 y 1)
    numeric_cols = [col for col in numeric_cols if not ((df[col].unique().len() == 2) & (df[col].is_in([0, 1]).sum() == df[col].len()))]

    # Aplicar Min-Max Scaler solo a las columnas numéricas no binarias
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        if range_val != 0:  # Evita la división por cero en caso de que todas las entradas en una columna sean iguales
            df = df.with_columns(
                ((pl.col(col) - min_val) / range_val).alias(col)
            )

    logger.info("Estandarización con Min-Max Scaler completada!")

    # Retorna el DataFrame estandarizado
    return df


# 2. Balance la target variable
def balance_target_variable_pl(
        df: pl.DataFrame,
        params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Balancea la variable objetivo utilizando el método Synthetic Minority Over-sampling Technique (SMOTE).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame de Polars que se balanceará la target.
    params: Dict[str, Any]
        Diccionario de parámetros model input.


    Returns
    -------
    pl.DataFrame
        DataFrame con target balanceada balanceado.
    """
    logger.info("Iniciando el balanceo de la variable objetivo con SMOTE...")

    # Parámetros
    target = params['target']
    random_state = params['balance_target_variable']['random_state']
    sampling_strategy = params['balance_target_variable']['sampling_strategy']

    # Separar las características y la variable objetivo
    X = df.drop(target[0])
    y = df[target]

    # Contar las clases antes del balanceo
    counts_before = y.groupby(target).count()
    logger.info(f"Conteo de clases antes del balanceo: {counts_before}")

    # Inicializar el objeto SMOTE
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)

    # Aplicar SMOTE
    X_resampled, y_resampled = smote.fit_resample(X.to_numpy(), y.to_numpy().flatten())

    # Crear un nuevo DataFrame con las características y la variable objetivo balanceada
    df_resampled = pl.DataFrame(X_resampled, schema=X.schema)
    df_resampled = df_resampled.with_columns(pl.Series(target[0], y_resampled))

    # Contar las clases después del balanceo
    counts_after = pl.DataFrame({target[0]: y_resampled}).groupby(target).agg(pl.count())
    logger.info(f"Conteo de clases después del balanceo: {counts_after}")

    logger.info("Balanceo de la variable objetivo con SMOTE completado!")

    # Retorna el DataFrame balanceado
    return df_resampled


# 3. Separación de datos en entrenamiento y prueba
def train_test_split_pl(
        df: pl.DataFrame,
        params: Dict[str, Any]
) -> (pl.DataFrame, pl.DataFrame):
    """
    Divide un DataFrame en dos subconjuntos: entrenamiento y prueba.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame de Polars que se dividirá.
    params: Dict[str, Any]
        Diccionario de parámetros model input.

    Returns
    -------
    (pl.DataFrame, pl.DataFrame)
        Tupla con los DataFrames de entrenamiento y prueba.
    """
    logger.info("Iniciando la división de datos en entrenamiento y prueba...")

    # Parámetros
    target = params['target'][0]
    test_size = params['train_test_split']['test_size']
    random_state = params['train_test_split']['random_state']
    shuffle = params['train_test_split']['shuffle']

    # Separar las características y la variable objetivo
    X = df.drop(target)
    y = df[target]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(),
                                                        y.to_numpy().flatten(),
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y,
                                                        shuffle=shuffle)

    # Crear DataFrames con los datos de entrenamiento y prueba
    train_data = pl.DataFrame(X_train, schema=X.schema)
    test_data = pl.DataFrame(X_test, schema=X.schema)

    # Convertir numpy arrays a Series de Polars y añadir como nuevas columnas
    y_train_series = pl.Series(name=target, values=y_train)
    y_test_series = pl.Series(name=target, values=y_test)

    train_data = train_data.with_columns(y_train_series)
    test_data = test_data.with_columns(y_test_series)

    logger.info("División de datos en entrenamiento y prueba completada!")

    # Retorna los DataFrames de entrenamiento y prueba
    return train_data, test_data