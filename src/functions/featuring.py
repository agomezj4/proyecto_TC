from typing import Dict, List, Any

import polars as pl
import pandas as pd
import logging
import re

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
        # pl.when(pl.col(campo[5]) <= limite_categoria_ultima_camp[0])
        # .then(pl.lit('reciente').alias(campo[6]))
        # .when(pl.col(campo[5]).is_between(limite_categoria_ultima_camp[0] + 1, limite_categoria_ultima_camp[1]))
        # .then(pl.lit('moderado').alias(campo[6]))
        # .otherwise(pl.lit('antiguo').alias(campo[6])),

        # Ingresos Per Cápita
        (pl.col(campo[1]) / pl.col(campo[7])).alias(campo[8]),

        # # Combinación de Perfil de Riesgo y Estado Civil
        # pl.concat_str([pl.col(campo[9]).cast(str), pl.col(campo[10])]).alias(campo[11]),
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
        Diccionario de parámetros featuring.

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


# 3. Encoding de variables categóricas
def cumulatively_categorise_pl(
        column: pl.Series,
        params: Dict[str, Any],
) -> pl.Series:
    """
    Categoriza acumulativamente una columna de un DataFrame de Polars, reemplazando los valores
    que no cumplen con el umbral especificado.

    Parameters
    ----------
    column : polars.Series
        Columna de un DataFrame de Polars que se categorizará acumulativamente.
    params: Dict[str, Any]
        Diccionario de parámetros featuring.

    Returns
    -------
    polars.Series
        Columna de un DataFrame de Polars con la categorización acumulativa aplicada.
    """
    logger.info(f"Empieza el proceso de categorización acumulativa para el campo '{column.name}'...")

    # Parámetros
    threshold = params['threshold']
    replacement_value = params['value']

    # Calculamos el valor de umbral basado en el porcentaje dado
    threshold_value = int(threshold * column.len())

    # Calculamos los conteos y ordenamos de forma descendente
    counts = column.to_frame().groupby(column).agg(pl.count()).sort(by=column.name, descending=True)

    # Acumulamos las frecuencias hasta llegar o superar el umbral
    counts = counts.with_columns(pl.col("count").cumsum().alias("cumulative_count"))
    valid_categories = counts.filter(pl.col("cumulative_count") <= threshold_value).select(column.name)

    # Creamos una lista con las categorías válidas más el valor de reemplazo
    valid_categories = valid_categories.to_series().to_list() + [replacement_value]

    # Reemplazamos los valores que no están en la lista de categorías válidas
    new_column = column.apply(lambda x: x if x in valid_categories else replacement_value, return_dtype=column.dtype)

    logger.info(f"Categorización acumulativa completada para el campo '{column.name}'!")

    return new_column


def replace_spaces_with_underscores(category):
    return re.sub(r'\s+', '_', category)


def one_hot_encoding_pl(
        df: pl.DataFrame,
        params: Dict[str, Any],
) -> pl.DataFrame:
    """
    Aplica One Hot Encoding a las columnas especificadas en el diccionario de parámetros.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame de polars al que se le aplicará One Hot Encoding.
    params: Dict[str, Any]
        Diccionario de parámetros featuring.

    Returns
    -------
    pl.DataFrame: DataFrame con las columnas transformadas.
    """
    logger.info("Iniciando One Hot Encoding...")

    # Parámetros
    cum_cat = params['cum_cat']
    one_hot_encoder = [nombre for nombre, dtype in df.schema.items() if dtype == pl.Utf8]

    for var in one_hot_encoder:
        # `cumulatively_categorise` es una función definida que categoriza y luego transforma en códigos enteros.
        df = df.with_columns(
            cumulatively_categorise_pl(df[var], cum_cat).alias(var)
        )

        # Realizando One Hot Encoding
        df = df.with_columns([
            pl.when(df[var] == category)
            .then(1)
            .otherwise(0)
            .alias(f"{var}_{replace_spaces_with_underscores(category)}")
            for category in df[var].unique().to_list()
        ]).drop(var)

    logger.info("One Hot Encoding completado!")

    return df


# 4. Selección de características
def add_random_variables_pd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega dos variables aleatorias al DataFrame

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame al que se agregarán las variables aleatorias.

    Returns
    -------
    pd.DataFrame
        DataFrame con las variables aleatorias agregadas.
    """
    # Establecer la semilla fija en 42
    np.random.seed(42)

    # Agregar variables aleatorias
    df["var_aleatoria_uniforme"] = np.random.rand(len(df))
    df["var_aleatoria_entera"] = np.random.randint(1, 5, size=len(df))

    return df

def random_forest_selection_pl(
        df: pl.DataFrame,
        params: Dict[str, Any],
) -> pl.DataFrame:
    """
    Entrena un modelo RandomForest y calcula la importancia de las características usando Polars.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame de Polars al que se le calculará la importancia de las características.
    params: Dict[str, Any]
        Diccionario de parámetros featuring.

    Returns
    -------
    pl.DataFrame: DataFrame con la importancia de las características calculadas.
    """

    logger.info("Iniciando la selección de características con Random Forest...")

    # Parámetros
    rf_selection = params['rf_selection']

    # Convertimos a pandas para usar train_test_split
    df_pandas = df.to_pandas()

    # Divide los datos en conjuntos de entrenamiento y prueba
    X = df_pandas.drop(columns=[rf_selection['target'], rf_selection['id']])
    y = df_pandas[rf_selection['target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=rf_selection['test_size'],
                                                        random_state=rf_selection['seed'])

    # Agregar las variables aleatorias al DataFrame
    X_train = add_random_variables_pd(X_train)

    # Crea un clasificador Bagging
    bag_class = BaggingClassifier(estimator=RandomForestClassifier(
        n_estimators=rf_selection['n_estimators'],
        max_depth=rf_selection['max_depth'],
        random_state=rf_selection['seed']))

    # Ajusta el modelo a tus datos de entrenamiento
    bag_class.fit(X_train, y_train)

    # Calcula la importancia de las características
    feature_importance = np.mean([
        tree.feature_importances_ for tree in bag_class.estimators_
    ], axis=0)

    # Obtén los nombres de las características
    feature_names = X_train.columns.tolist()

    # Crea un DataFrame para mostrar la importancia de las características
    feature_importance_df = pl.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    })

    # Ordena el DataFrame por importancia de manera descendente
    feature_importance_df = feature_importance_df.sort("Importance", descending=True)

    # Obtener la importancia de las variables aleatorias
    random_var_imp_0_1 = feature_importance_df.filter(pl.col("Feature") == "var_aleatoria_uniforme").select("Importance").to_numpy()[0][0]
    random_var_imp_1_4 = feature_importance_df.filter(pl.col("Feature") == "var_aleatoria_entera").select("Importance").to_numpy()[0][0]

    # Eliminar las variables con importancia menor que las variables aleatorias
    feature_importance_df = feature_importance_df.filter(
        (pl.col("Importance") > random_var_imp_0_1) & (pl.col("Importance") > random_var_imp_1_4))

    logger.info("Selección de características con Random Forest completada!")

    return feature_importance_df


def entropy(p):
    """
    Calcula la entropía de un conjunto de probabilidades.

    Parameters
    ----------
    p : list
        Lista de probabilidades.

    Returns
    -------
    float: Valor de la entropía.
    """
    return -np.sum([pi * np.log2(pi) if pi > 0 else 0 for pi in p])


def conditional_entropy_selection_pl(
        df: pl.DataFrame,
        params: Dict[str, Any],
) -> pl.DataFrame:
    """
    Calcula la entropía condicional y la ganancia de información para un
    conjunto de variables usando polars.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame de polars al que se le calculará la entropía condicional.
    params: Dict[str, Any]
        Diccionario de parámetros featuring.

    Returns
    -------
    pl.DataFrame: DataFrame con la entropía condicional y la ganancia de
    información calculadas.
    """
    logger.info("Iniciando la selección de características con Entropía Condicional...")

    # Parámetros
    target = params['ce_selection']['target']
    id = params['ce_selection']['id']

    # Eliminar id y target
    df = df.drop(id)

    # Entropía variable objetivo
    des = df.group_by(target).agg(pl.len().alias('count'))
    pr = des['count'] / df.height  # Obtener el número total de filas con df.height()
    Ho = entropy(pr.to_numpy())

    # Convertir df a pandas para agregar las variables aleatorias
    df_pandas = df.to_pandas()
    df_pandas = add_random_variables_pd(df_pandas)

    # Convertir de nuevo a DataFrame de Polars
    df = pl.DataFrame(df_pandas)

    # Cálculo de entropía condicional
    feature_names, feature_importance = [], []

    for columna in df.columns:
        if columna == target:
            continue
        H = 0
        feature_names.append(columna)

        # Cálculo de entropía condicional en el bucle for
        grouped = df.group_by(columna).agg(pl.col(target).count().alias("count"))
        # Aseguramos que cada columna sea accesible por su nombre para evitar confusiones futuras
        grouped = grouped.with_columns(pl.col(columna).alias('value'))
        for row in grouped.rows():
            # Usamos índices de acuerdo a cómo están ordenadas las columnas en el DataFrame agrupado
            value, group_count = row[0], row[1]  # Asumiendo que 'columna' y 'count' son las primeras dos columnas
            df_i = df.filter(pl.col(columna) == value)
            des = df_i.group_by(target).agg(pl.len().alias('count'))
            pr = des['count'] / group_count
            Hcond = entropy(pr.to_numpy())
            prob = group_count / df.height  # Usamos la propiedad height de Polars directamente
            H += Hcond * prob

        feature_importance.append(Ho - H)

    # Crear un DataFrame con la importancia de las características
    data_entropia = pl.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

    # Ordenar el DataFrame por importancia de manera descendente
    df_entropia = data_entropia.sort(by='Importance', descending=True)

    # Obtener la importancia de las variables aleatorias
    random_var_imp_0_1 = \
    df_entropia.filter(pl.col("Feature") == "var_aleatoria_uniforme").select("Importance").to_numpy()[0][0]
    random_var_imp_1_4 = \
    df_entropia.filter(pl.col("Feature") == "var_aleatoria_entera").select("Importance").to_numpy()[0][0]

    # Eliminar las variables con importancia menor que las variables aleatorias
    df_entropia = df_entropia.filter(
        (pl.col("Importance") > random_var_imp_0_1) & (pl.col("Importance") > random_var_imp_1_4))

    logger.info("Selección de características con Entropía Condicional completada!")

    return df_entropia


def intersect_top_features_pl(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    params: Dict[str, Any],
) -> set:
    """
    Obtiene las características más importantes de dos DataFrames basado en un diccionario de parámetros

    Parameters
    ----------
    df1, df2: polars.DataFrame
        DataFrames de polars de los que se obtendrán las características más importantes
    params: Dict[str, Any]
        Diccionario de parámetros featuring.

    Returns
    -------
    set: Conjunto con las características más importantes.
    """
    logger.info("Intersección de las feature importence por los métodos aplicados...")

    # Parámetros
    n = params['features_importance']['top_features']

    # Obtén las n características más importantes del dataframe
    top_features_df1 = set(df1.select('Feature').limit(n).to_pandas()['Feature'])
    top_features_df2 = set(df2.select('Feature').limit(n).to_pandas()['Feature'])

    # Intersecta los conjuntos para obtener las características comunes a los dos dataframes
    top_features = top_features_df1.intersection(top_features_df2)

    logger.info("Intersección de las feature importence completada!")

    # Retorna el conjunto con las características más importantes
    return top_features


# validar que en el top_features no esta id ni taget. toma el df de one hot encoding y filtra solo las columnas de top_features + target.