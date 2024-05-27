from typing import Dict, Any


import pandas as pd
import logging
import re
import numpy as np

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 1. Nuevas Variables
def new_features_pd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Genera nuevas características en un DataFrame de Pandas basándose en los parámetros especificados,
    como ratios financieros y categorías derivadas de datos existentes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de Pandas sobre el cual se añadirán nuevas características
    params: Dict[str, Any]
        Diccionario de parámetros featuring

    Returns
    -------
    pd.DataFrame
        DataFrame con las nuevas características añadidas.
    """
    logger.info("Iniciando la generación de nuevas características...")

    # Parámetros
    campo = params['campos_new_features']
    limite_categoria_ultima_camp = params['limite_ultima_camp']
    categoria_ultima_camp = params['categorias_ultima_camp']

    # Ratio de Ingresos a Deuda
    df[campo[2]] = df[campo[1]] / df[campo[0]].replace({0: float('inf')})

    # Ratio de Deuda a Créditos
    df[campo[4]] = df[campo[0]] / df[campo[3]].replace({0: float('inf')})

    # Categorización de los días desde la última campaña
    # df[campo[6]] = pd.cut(
    #     df[campo[5]],
    #     bins=[-float('inf'), limite_categoria_ultima_camp[0], limite_categoria_ultima_camp[1], float('inf')],
    #     labels=['reciente', 'moderado', 'antiguo']
    # )

    # Ingresos Per Cápita
    df[campo[8]] = df[campo[1]] / df[campo[7]]

    # Combinación de Perfil de Riesgo y Estado Civil
    # df[campo[11]] = df[campo[9]].astype(str) + df[campo[10]]

    logger.info("Nuevas características generadas!")

    return df


# 2. Agregar la variable objetivo al DataFrame
def add_target_variable_pd(df1: pd.DataFrame, df2: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Agrega la variable objetivo al DataFrame de Pandas.

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame de Pandas al cual se añadirá la variable objetivo
    df2 : pd.DataFrame
        DataFrame de Pandas que contiene la variable objetivo
    params: Dict[str, Any]
        Diccionario de parámetros featuring

    Returns
    -------
    pd.DataFrame
        DataFrame con la variable objetivo añadida.
    """
    logger.info("Añadiendo la variable objetivo al DataFrame...")

    # Parámetros
    target_params = params['general']

    # Realizar un left join para añadir la columna 'y' de df2 a df1 basado en 'id'
    df = df1.merge(df2[[target_params[0], target_params[1]]], on=target_params[0], how='left')

    logger.info("Variable objetivo añadida!")

    return df


# 3. Encoding de variables categóricas
def cumulatively_categorise_pd(column: pd.Series, params: Dict[str, Any]) -> pd.Series:
    """
    Categoriza acumulativamente una columna de un DataFrame de Pandas, reemplazando los valores
    que no cumplen con el umbral especificado.

    Parameters
    ----------
    column : pd.Series
        Columna de un DataFrame de Pandas que se categorizará acumulativamente
    params: Dict[str, Any]
        Diccionario de parámetros featuring

    Returns
    -------
    pd.Series
        Columna de un DataFrame de Pandas con la categorización acumulativa aplicada.
    """
    logger.info(f"Empieza el proceso de categorización acumulativa para el campo '{column.name}'...")

    # Parámetros
    threshold = params['threshold']
    replacement_value = params['value']

    # Calculamos el valor de umbral basado en el porcentaje dado
    threshold_value = int(threshold * len(column))

    # Calculamos los conteos y ordenamos de forma descendente
    counts = column.value_counts().sort_values(ascending=False)

    # Acumulamos las frecuencias hasta llegar o superar el umbral
    cumulative_counts = counts.cumsum()
    valid_categories = cumulative_counts[cumulative_counts <= threshold_value].index.tolist()

    # Creamos una lista con las categorías válidas más el valor de reemplazo
    valid_categories.append(replacement_value)

    # Reemplazamos los valores que no están en la lista de categorías válidas
    new_column = column.apply(lambda x: x if x in valid_categories else replacement_value)

    logger.info(f"Categorización acumulativa completada para el campo '{column.name}'!")

    return new_column


def replace_spaces_with_underscores(category):
    return re.sub(r'\s+', '_', category)


def one_hot_encoding_pd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Aplica One Hot Encoding a las columnas especificadas en el diccionario de parámetros.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de Pandas al que se le aplicará One Hot Encoding
    params: Dict[str, Any]
        Diccionario de parámetros featuring

    Returns
    -------
    pd.DataFrame: DataFrame con las columnas transformadas.
    """
    logger.info("Iniciando One Hot Encoding...")

    # Parámetros
    cum_cat = params['cum_cat']
    one_hot_encoder = [nombre for nombre in df.columns if df[nombre].dtype == 'object']

    for var in one_hot_encoder:
        # `cumulatively_categorise` es una función definida que categoriza y luego transforma en códigos enteros.
        df[var] = cumulatively_categorise_pd(df[var], cum_cat)

        # Realizando One Hot Encoding
        dummies = pd.get_dummies(df[var], prefix=var, prefix_sep='_')
        df = pd.concat([df.drop(columns=[var]), dummies], axis=1)

    logger.info("One Hot Encoding completado!")

    return df


# 4. Selección de características
def add_random_variables_pd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega dos variables aleatorias al DataFrame

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame al que se agregarán las variables aleatorias

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


def random_forest_selection_pd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Entrena un modelo RandomForest y calcula la importancia de las características usando Pandas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de Pandas al que se le calculará la importancia de las características
    params: Dict[str, Any]
        Diccionario de parámetros featuring

    Returns
    -------
    pd.DataFrame: DataFrame con la importancia de las características calculadas.
    """
    logger.info("Iniciando la selección de características con Random Forest...")

    # Parámetros
    rf_selection = params['rf_selection']
    id_target = params['general']

    # Divide los datos en conjuntos de entrenamiento y prueba
    X = df.drop(columns=[id_target[1], id_target[0]])
    y = df[id_target[1]]
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
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    })

    # Ordena el DataFrame por importancia de manera descendente
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    # Obtener la importancia de las variables aleatorias
    random_var_imp_0_1 = feature_importance_df.loc[feature_importance_df['Feature'] == 'var_aleatoria_uniforme', 'Importance'].values[0]
    random_var_imp_1_4 = feature_importance_df.loc[feature_importance_df['Feature'] == 'var_aleatoria_entera', 'Importance'].values[0]

    # Eliminar las variables con importancia menor que las variables aleatorias
    feature_importance_df = feature_importance_df[
        (feature_importance_df["Importance"] > random_var_imp_0_1) &
        (feature_importance_df["Importance"] > random_var_imp_1_4)
    ]

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


def conditional_entropy_selection_pd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Calcula la entropía condicional y la ganancia de información para un
    conjunto de variables usando Pandas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de Pandas al que se le calculará la entropía condicional
    params: Dict[str, Any]
        Diccionario de parámetros featuring

    Returns
    -------
    pd.DataFrame: DataFrame con la entropía condicional y la ganancia de
    información calculadas.
    """
    logger.info("Iniciando la selección de características con Entropía Condicional...")

    # Parámetros
    id_target = params['general']
    threshold = params['ce_selection']['threshold']

    # Eliminar id y target
    df = df.drop(columns=[id_target[0]])

    # Entropía variable objetivo
    target_counts = df[id_target[1]].value_counts()
    pr = target_counts / len(df)
    Ho = entropy(pr)

    # Cálculo de entropía condicional
    feature_names, feature_importance = [], []

    for column in df.columns:
        if column == id_target[1]:
            continue
        H = 0
        feature_names.append(column)

        # Cálculo de entropía condicional en el bucle for
        grouped = df.groupby(column)[id_target[1]].value_counts().unstack(fill_value=0)
        for value, group_counts in grouped.iterrows():
            pr = group_counts / group_counts.sum()
            Hcond = entropy(pr)
            prob = group_counts.sum() / len(df)
            H += Hcond * prob

        feature_importance.append(Ho - H)

    # Crear un DataFrame con la importancia de las características
    data_entropia = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

    # Ordenar el DataFrame por importancia de manera descendente
    df_entropia = data_entropia.sort_values(by='Importance', ascending=False)

    # Filtrar características según el umbral de importancia
    df_entropia = df_entropia[df_entropia['Importance'] >= threshold]

    logger.info("Selección de características con Entropía Condicional completada!")

    return df_entropia


def intersect_top_features_pd(
    df1: pd.DataFrame, # Características de random forest
    df2: pd.DataFrame, # Características de entropía
    df3: pd.DataFrame, # Dataframe original
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Obtiene las características más importantes de dos DataFrames basado en un diccionario de parámetros.

    Parameters
    ----------
    df1, df2, df3: pd.DataFrame
        DataFrames de Pandas de los que se obtendrán las características más importantes y el dataframe original
    params: Dict[str, Any]
        Diccionario de parámetros featuring

    Returns
    -------
    pd.DataFrame: DataFrame con las características más importantes más la variable objetivo.
    """
    logger.info("Intersección de las feature importance por los métodos aplicados...")

    # Parámetros
    n = params['features_importance']['top_features']
    id_target = params['general']

    # Obtén las n características más importantes del dataframe
    top_features_df1 = set(df1['Feature'].head(n))
    top_features_df2 = set(df2['Feature'].head(n))

    # Une los conjuntos para obtener todas las características únicas de los dos dataframes
    top_features = top_features_df1.union(top_features_df2)

    logger.info("Intersección de las feature importance completada!")

    # Añade la variable objetivo y filtra el dataframe original
    top_features.add(id_target[1])
    filtered_df = df3[list(top_features.intersection(df3.columns))]

    # Asegúrate de excluir la columna 'id', si existe
    if id_target[0] in filtered_df.columns:
        filtered_df = filtered_df.drop(columns=[id_target[0]])

    logger.info("DataFrame con las características más importantes y la variable objetivo generado!")

    return filtered_df


