from typing import Dict, Any, List


import pandas as pd
import logging
import os
import sys
import numpy as np

from sklearn.pipeline import Pipeline

# Ajustar el path del sistema para incluir el directorio 'src' al nivel superior
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#1. Validacación de datos.
def validate_data(df: pd.DataFrame, tag_dict: pd.DataFrame) -> pd.DataFrame:
    """
    Valida las etiquetas y tipos de datos del DataFrame basado en el diccionario de etiquetas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    tag_dict : pd.DataFrame
        Diccionario de etiquetas.

    Returns
    -------
    pd.DataFrame
        DataFrame validado.
    """
    logger.info("Validando etiquetas y tipos de datos...")
    from functions.processing import validate_tags_pd, validate_dtypes_pd

    try:
        data_validate_tags = validate_tags_pd(df, tag_dict)
        data_validate_dtypes = validate_dtypes_pd(data_validate_tags, tag_dict)
    except Exception as e:
        logger.error(f"Error en la validación del conjunto de datos: {e}")
        raise

    return data_validate_dtypes


#2. Procesamiento de datos.
def process_data(df: pd.DataFrame, tag_dict: pd.DataFrame, params_processing: Dict[str, Any]) -> pd.DataFrame:
    """
    Procesa el DataFrame aplicando varios pasos de limpieza y transformación.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    tag_dict : pd.DataFrame
        Diccionario de etiquetas.
    params_processing : Dict[str, Any]
        Parámetros para el procesamiento de datos.

    Returns
    -------
    pd.DataFrame
        DataFrame procesado.
    """
    logger.info("Procesando el conjunto de datos...")
    from functions.processing import change_names_pd, change_dtype_pd, delete_accents_pd
    from functions.processing import standardize_binary_values_pd, impute_missing_values_pd

    try:
        data_change_names = change_names_pd(df, tag_dict)
        data_change_dtype = change_dtype_pd(data_change_names, tag_dict)
        data_delete_accents = delete_accents_pd(data_change_dtype)
        data_standardize_binary_values = standardize_binary_values_pd(data_delete_accents, params_processing)
        data_processing = impute_missing_values_pd(data_standardize_binary_values)
    except Exception as e:
        logger.error(f"Error en el procesamiento del conjunto de datos: {e}")
        raise

    return data_processing


#3. Ingeniería de características.
def feature_engineering(df: pd.DataFrame, params_featuring: Dict[str, Any], features: List[str]) -> pd.DataFrame:
    """
    Aplica ingeniería de características y selecciona las características especificadas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    params_featuring : Dict[str, Any]
        Parámetros para la ingeniería de características.
    features : List[str]
        Lista de características seleccionadas.

    Returns
    -------
    pd.DataFrame
        DataFrame con las características seleccionadas.
    """
    logger.info("Aplicando ingeniería de características...")
    from functions.featuring import new_features_pd, one_hot_encoding_pd

    try:
        data_features = new_features_pd(df, params_featuring)
        data_one_hot = one_hot_encoding_pd(data_features, params_featuring)
        data_selected_features = data_one_hot[features]
    except Exception as e:
        logger.error(f"Error en el feature engineering: {e}")
        raise

    return data_selected_features


#4. Preparación de datos para el modelo.
def prepare_data_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el DataFrame para el modelo escalando las características.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las características seleccionadas.

    Returns
    -------
    pd.DataFrame
        DataFrame preparado para el modelo.
    """
    logger.info("Preparando los datos para el modelo...")
    from functions.model_input import min_max_scaler_pd

    try:
        data_scaled = min_max_scaler_pd(df)
    except Exception as e:
        logger.error(f"Error en la preparación de datos para el modelo: {e}")
        raise

    return data_scaled


#.5 Predicción.
def generate_predictions(model: Pipeline, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Genera predicciones y probabilidades utilizando el modelo entrenado.

    Parameters
    ----------
    model : Pipeline
        Modelo entrenado.
    df1 : pd.DataFrame
        DataFrame preparado para el modelo.
    df2 : pd.DataFrame
        DataFrame procesado para el modelo.

    Returns
    -------
    pd.DataFrame
        DataFrame con las predicciones y probabilidades añadidas.
    """
    logger.info("Generando predicciones con el modelo entrenado...")
    try:
        predictions = model.predict(df1)
        probs = model.predict_proba(df1)
    except Exception as e:
        logger.error(f"Error en la generación de predicciones: {e}")
        raise

    df2['acepta_tc'] = predictions
    df2['proba_acepta_tc'] = np.round(probs[:, 1], 2)
    df_final = df2[['id_cliente',
                                'acepta_tc',
                                'proba_acepta_tc']]

    return df_final


#6. Agregar predicciones al DataFrame.
def add_predictions(df: pd.DataFrame,
                    tag_dict: pd.DataFrame,
                    params_processing: Dict[str, Any],
                    params_featuring: Dict[str, Any],
                    model: Pipeline,
                    features: List[str]) -> pd.DataFrame:
    """
    Califica un conjunto de datos basado en un modelo previamente entrenado y una lista de variables seleccionadas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de Pandas que contiene los datos de entrada.
    tag_dict : pd.DataFrame
        Diccionario de etiquetas.
    params_processing: Dict[str, Any]
        Diccionario de parámetros de procesamiento.
    params_featuring: Dict[str, Any]
        Diccionario de parámetros de ingeniería de características.
    model : Pipeline
        Modelo entrenado que se usará para hacer las predicciones.
    features : List[str]
        Lista de nombres de las columnas en el DataFrame que se usarán como características para el modelo.

    Returns
    -------
    pd.DataFrame
        DataFrame original con una nueva columna de predicciones añadida.
    """
    logger.info("Iniciando la adición de predicciones al conjunto de datos...")

    data_validated = validate_data(df, tag_dict)
    data_processed = process_data(data_validated, tag_dict, params_processing)
    data_features = feature_engineering(data_processed, params_featuring, features)
    data_prepared = prepare_data_for_model(data_features)
    df_final = generate_predictions(model, data_prepared, data_processed)

    logger.info("Predicciones añadidas al DataFrame.")

    return df_final
