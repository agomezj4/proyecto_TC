from typing import Dict, List, Any

import polars as pl
import pandas as pd
import logging

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 1. Nuevas Variables
def new_features_pl(
        df: pl.DataFrame,
        params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Genera nuevas características en un DataFrame de Polars basándose en los parámetros especificados,
    como ratios financieros y categorías derivadas de datos existentes.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame de polars sobre el cual se añadirán nuevas características.
    params: Dict[str, Any]
        Diccionario de parámetros processing.

    Returns
    -------
    pl.DataFrame
        DataFrame con las nuevas características añadidas.
    """
    logger.info("Iniciando la generación de nuevas características...")

    # Parámetros
    campo = params['campos_new_features']
    limite_categoria_ultima_camp = params['limite_ultima_camp']
    categoria_ultima_camp = params['categorias_ultima_camp']

    # Nuevas transformaciones según los parámetros
    new_columns = [
        # Ratio de Ingresos a Deuda
        pl.when(pl.col(campo[0]) == 0).then(float('inf'))  # Evitar división por cero
        .otherwise(pl.col(campo[1]) / pl.col(campo[0])).alias(campo[2]),

        # Ratio de Deuda a Créditos
        pl.when(pl.col(campo[3]) == 0).then(float('inf'))  # Evitar división por cero
        .otherwise(pl.col(campo[0]) / pl.col(campo[3])).alias(campo[4]),

        # Categorización de los días desde la última campaña
        pl.when(pl.col(campo[5]) <= limite_categoria_ultima_camp[0])
        .then(pl.lit('reciente').alias(campo[6]))
        .when(pl.col(campo[5]).is_between(limite_categoria_ultima_camp[0] + 1, limite_categoria_ultima_camp[1]))
        .then(pl.lit('moderado').alias(campo[6]))
        .otherwise(pl.lit('antiguo').alias(campo[6])),

        # Ingresos Per Cápita
        (pl.col(campo[1]) / pl.col(campo[7])).alias(campo[8]),

        # Combinación de Perfil de Riesgo y Estado Civil
        pl.concat_str([pl.col(campo[9]).cast(str), pl.col(campo[10])]).alias(campo[11]),
    ]

    # Aplicar todas las transformaciones en una sola operación
    df = df.with_columns(new_columns)

    logger.info("Nuevas características generadas!")

    return df


# 2. Agregar la variable objetivo al DataFrame
def add_target_variable_pl(
        df1: pl.DataFrame, #df
        df2: pl.DataFrame, #target
        params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Agrega la variable objetivo al DataFrame de Polars.

    Parameters
    ----------
    df1 : polars.DataFrame
        DataFrame de polars al cual se añadirá la variable objetivo.
    df12 : polars.DataFrame
        DataFrame de polars que contiene la variable objetivo.
    params: Dict[str, Any]
        Diccionario de parámetros processing.

    Returns
    -------
    pl.DataFrame
        DataFrame con la variable objetivo añadida.
    """
    logger.info("Añadiendo la variable objetivo al DataFrame...")

    # Parámetros
    target_params = params['target']

    # Realizar un left join para añadir la columna 'y' de df2 a df1 basado en 'id'
    merged_df = df1.join(df2.select([target_params[0], target_params[1]]), on=target_params[0], how='left')

    # Renombrar la columna si es necesario o realizar cualquier transformación adicional
    df = merged_df.rename({target_params[1]: target_params[2]})

    logger.info("Variable objetivo añadida!")

    return df


# 3. Selección de características
def random_forest_selection(
        df: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Entrena un modelo RandomForest y calcula la importancia de las características

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de pandas al que se le calculará la importancia de las características
    params: Dict[str, Any]
        Diccionario de parámetros

    Returns
    -------
    pd.DataFrame: DataFrame con la importancia de las características calculadas
    """

    # Divide los datos en conjuntos de entrenamiento y prueba
    X = df.drop(columns=rf_selection['target'])
    y = df[rf_selection['target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=rf_selection['test_size'],
                                                        random_state=rf_selection['seed'])

    # Agregar las variables aleatorias al DataFrame
    X_train = add_random_variables(X_train)

    # Crea un clasificador Bagging
    bag_class = BaggingClassifier(estimator=RandomForestClassifier(    n_estimators=rf_selection['n_estimators'],
        max_depth=rf_selection['max_depth'],
        random_state=rf_selection['seed'])


    # Ajusta el modelo a tus datos de entrenamiento
    bag_class.fit(X_train, y_train)

    # Calcula la importancia de las características
    feature_importance = bag_class.feature_importances_

    # Obtén los nombres de las características
    feature_names = X_train.columns

    # Crea un DataFrame para mostrar la importancia de las características
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importance}
    )

    # Ordena el DataFrame por importancia de manera descendente
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    # Obtener la importancia de las variables aleatorias
    random_var_imp_0_1 = \
    feature_importance_df.loc[feature_importance_df["Feature"] == "var_aleatoria_uniforme", "Importance"].values[0]
    random_var_imp_1_4 = \
    feature_importance_df.loc[feature_importance_df["Feature"] == "var_aleatoria_entera", "Importance"].values[0]

    # Eliminar las variables con importancia menor que las variables aleatorias
    feature_importance_df = feature_importance_df[(feature_importance_df["Importance"] > random_var_imp_0_1) & (
                feature_importance_df["Importance"] > random_var_imp_1_4)]

    return feature_importance_df
