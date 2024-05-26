from typing import Dict, Any, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
import logging

from sklearn.model_selection import train_test_split

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 1. Escalado de variables numéricos
def min_max_scaler_pd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza las columnas numéricas (excluyendo binarias) de un DataFrame utilizando el método Min-Max Scaler.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de Pandas que se estandarizará

    Returns
    -------
    pd.DataFrame
        DataFrame estandarizado.
    """
    logger.info("Iniciando la estandarización con Min-Max Scaler...")

    # Identificar las columnas numéricas
    numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns

    # Filtrar solo las columnas numéricas no binarias (excluyendo aquellas que solo toman valores 0 y 1)
    numeric_cols = [col for col in numeric_cols if not ((df[col].nunique() == 2) & (df[col].isin([0, 1]).sum() == len(df)))]

    # Aplicar Min-Max Scaler solo a las columnas numéricas no binarias
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        if range_val != 0:  # Evita la división por cero en caso de que todas las entradas en una columna sean iguales
            df[col] = (df[col] - min_val) / range_val

    logger.info("Estandarización con Min-Max Scaler completada!")

    return df


# 2. Balance la target variable
def balance_target_variable_pd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Balancea la variable objetivo utilizando el método Synthetic Minority Over-sampling Technique (SMOTE).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de Pandas que se balanceará la target
    params: Dict[str, Any]
        Diccionario de parámetros model input

    Returns
    -------
    pd.DataFrame
        DataFrame con target balanceada balanceado.
    """
    logger.info("Iniciando el balanceo de la variable objetivo con SMOTE...")

    # Parámetros
    target = params['target'][0]
    random_state = params['balance_target_variable']['random_state']
    sampling_strategy = params['balance_target_variable']['sampling_strategy']

    # Separar las características y la variable objetivo
    X = df.drop(columns=[target])
    y = df[target]

    # Contar las clases antes del balanceo
    counts_before = y.value_counts()
    logger.info(f"Conteo de clases antes del balanceo: {counts_before}")

    # Inicializar el objeto SMOTE
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)

    # Aplicar SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Crear un nuevo DataFrame con las características y la variable objetivo balanceada
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target] = y_resampled

    # Contar las clases después del balanceo
    counts_after = y_resampled.value_counts()
    logger.info(f"Conteo de clases después del balanceo: {counts_after}")

    logger.info("Balanceo de la variable objetivo con SMOTE completado!")

    return df_resampled


# 3. Separación de datos en entrenamiento, validcaion y prueba
def train_test_split_pd(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide un DataFrame en tres subconjuntos: entrenamiento, validación y prueba.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de Pandas que se dividirá
    params: Dict[str, Any]
        Diccionario de parámetros model input

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tupla con los DataFrames de entrenamiento, validación y prueba.
    """
    logger.info("Iniciando la división de datos en entrenamiento, validación y prueba...")

    # Parámetros
    target = params['target'][0]
    test_size = params['train_test_split']['test_size']
    validation_size = params['train_test_split']['validation_size']
    random_state = params['train_test_split']['random_state']
    shuffle = params['train_test_split']['shuffle']

    # Separar las características y la variable objetivo
    X = df.drop(columns=[target])
    y = df[target]

    # Dividir los datos en entrenamiento y prueba
    X_train_val, X_test, y_train_val, y_test = train_test_split(X,
                                                                y,
                                                                test_size=test_size,
                                                                random_state=random_state,
                                                                stratify=y,
                                                                shuffle=shuffle)

    # Calcular el tamaño del conjunto de validación respecto al conjunto de entrenamiento + validación
    val_size = validation_size / (1 - test_size)

    # Dividir los datos de entrenamiento en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                      y_train_val,
                                                      test_size=val_size,
                                                      random_state=random_state,
                                                      stratify=y_train_val,
                                                      shuffle=shuffle)

    # Crear DataFrames con los datos de entrenamiento, validación y prueba
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    logger.info("División de datos en entrenamiento, validación y prueba completada!")

    return train_data, val_data, test_data

