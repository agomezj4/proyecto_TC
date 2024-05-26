from typing import Dict, Any

import pandas as pd
import unicodedata
import re
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 1. Validar tag de las fuentes
def validate_tags_pd(df: pd.DataFrame, tag_dict: pd.DataFrame) -> pd.DataFrame:
    """
    Valida que el número de tags identificados en source como raw sea igual al
    número de columnas en el dataframe utilizando Pandas.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe que contiene los datos a validar
    tag_dict : pd.DataFrame
        Diccionario de etiquetas

    Returns
    -------
    pd.DataFrame
        Si las tags están validadas, retorna el DataFrame original. En otros casos,
        solo emite un log
    """

    # Registra un mensaje de información indicando el inicio del proceso de validación de tags
    logger.info("Iniciando la validación de tags...")

    # Calcula la cantidad de tags identificados como 'raw' en el tag dictionary
    len_raw = tag_dict[tag_dict["source"] == "raw"].shape[0]

    # Comprueba si el número de tags es mayor que el número de columnas en el DataFrame
    if len_raw != df.shape[1]:
        raise ValueError(
            "Tags faltantes en el dataframe" if len_raw > df.shape[1] else "Tags faltantes en el tag dictionary")

    # Si el número de tags es igual al número de columnas, emite un registro informativo y retorna el DataFrame 'df'
    logger.info("Tags validados!")

    return df



# 2. Validar tipos de datos
def validate_dtypes_pd(df: pd.DataFrame, tag_dict: pd.DataFrame) -> pd.DataFrame:
    """
    Revisa que la tipología de los datos en el dataframe sea la misma estipulada
    en el tag_dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe que se quiere validar
    tag_dict : pd.DataFrame
        Diccionario de etiquetas

    Returns
    -------
    pd.DataFrame
        Si las tipologías de datos están validadas, retorna el DataFrame original.
        En otros casos, solo emite un log
    """

    logger.info("Iniciando la validación de tipos de datos...")

    # Filtra las filas del tag dictionary donde "source" es "raw"
    tag_dict_raw = tag_dict[tag_dict["source"] == "raw"]

    # Crear un diccionario de los tipos de datos esperados para cada columna, normalizando a minúsculas y eliminando espacios
    expected_types = {row['tag']: row['data_type'].strip().lower() for row in tag_dict_raw.to_dict(orient='records')}

    # Genera una lista de problemas con las diferencias entre los tipos de datos en el DataFrame y el tag dictionary
    problems = [
        f"{col} is {df[col].dtype} but should be {expected_types[col]}"
        for col in df.columns if col in expected_types and str(df[col].dtype).strip().lower() != expected_types[col]
    ]

    # Comprueba si se encontraron problemas y lanza una excepción TypeError si es así
    if problems:
        error_message = f"Se encontraron los siguientes problemas: {'/n'.join(problems)}"
        logger.error(error_message)
        raise TypeError(error_message)

    logger.info("Tipos de datos validados correctamente!")

    return df


# 3. Re- nombrar
def change_names_pd(df: pd.DataFrame, tag_dict: pd.DataFrame) -> pd.DataFrame:
    """
    Cambia los nombres de las columnas de un DataFrame según un tag dictionary y devuelve
    un nuevo DataFrame con los nombres cambiados utilizando Pandas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de Pandas del cual se cambiarán los nombres de las columnas
    tag_dict: pd.DataFrame
        Diccionario de etiquetas

    Returns
    -------
    pd.DataFrame: Un nuevo DataFrame con las columnas renombradas según el tag dictionary
    """
    logger.info("Iniciando el cambio de nombres...")

    # Filtra las filas del tag dictionary donde "source" sea "raw" y selecciona las columnas "tag" y "name"
    tag_dict_filtered = tag_dict[tag_dict["source"] == "raw"][["tag", "name"]]

    # Crea un diccionario de mapeo para cambiar los nombres de las columnas
    col_mapping = dict(zip(tag_dict_filtered['tag'], tag_dict_filtered['name']))

    # Cambia los nombres de las columnas según el tag dictionary "raw"
    df = df.rename(columns=col_mapping)

    logger.info("Nombres de columnas cambiados!")

    return df


# 4. Cambiar tipos de datos
def change_dtype_pd(df: pd.DataFrame, tag_dict: pd.DataFrame) -> pd.DataFrame:
    """
    Cambia el tipo de datos de cada columna en un DataFrame al tipo de datos especificado
    en el tag dictionary usando Pandas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de Pandas del cual se cambiarán los tipos de datos
    tag_dict: pd.DataFrame
        Diccionario de etiquetas

    Returns
    -------
    pd.DataFrame: DataFrame con las columnas cambiadas al nuevo tipo de datos.
    """
    logger.info("Iniciando el cambio de tipos de datos...")

    # Filtrar tag_dict para incluir solo las filas con "source" igual a "raw"
    tag_dict = tag_dict[tag_dict["source"] == "raw"]

    # Crear un diccionario de mapeo de tipos de datos
    type_mapping = dict(zip(tag_dict['name'], tag_dict['data_type_new']))

    # Cambiar el tipo de datos de cada columna según el mapeo
    for col in df.columns:
        if col in type_mapping:
            try:
                new_type = type_mapping[col]
                df[col] = df[col].astype(new_type)
            except Exception as e:
                logger.error(f"Error al intentar castear la columna {col} a {new_type}: {e}")

    logger.info("Dtypes cambiados!")

    return df


# 5. Eliminar acentos
def delete_accents_pd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina los acentos de las columnas identificadas como "str" de un DataFrame de Pandas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de Pandas del cual se eliminarán los acentos

    Returns
    -------
    pd.DataFrame
        DataFrame de Pandas con los acentos eliminados de las columnas especificadas.
    """
    logger.info("Iniciando la eliminación de acentos...")

    # Función para eliminar acentos
    def remove_accents(input_str):
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        return "".join([char for char in nfkd_form if not unicodedata.combining(char)])

    # Aplicar la eliminación de acentos solo a las columnas de tipo string
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: remove_accents(x) if isinstance(x, str) else x)

    logger.info("Acentos eliminados!")

    return df

# 6. Re categorizar columnas
def standardize_binary_values_pd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Estandariza valores en columnas específicas de un DataFrame de Pandas, transformando
    las diferentes formas de "sí" y "no" en valores binarios 1 y 0, respectivamente, y transformando
    los nulos o cadenas vacías a 0 en columnas específicas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de Pandas sobre el cual se realizarán las transformaciones
    params: Dict[str, Any]
        Diccionario de parámetros processing

    Returns
    -------
    pd.DataFrame: DataFrame con las columnas transformadas a valores binarios.
    """
    logger.info("Iniciando la estandarización de valores binarios...")

    # Columnas a transformar
    columns_to_transform1 = params['campos_binarios1']
    columns_to_transform2 = params['campos_binarios2']

    # Expresiones regulares comunes para "sí" y "no"
    yes_regex = re.compile(r'\bs[ií]|\byes\b', re.IGNORECASE)  # Regex para capturar variaciones de "sí"
    no_regex = re.compile(r'\bno\b', re.IGNORECASE)            # Regex para capturar variaciones de "no"

    # Función para transformar los valores de acuerdo a las expresiones regulares
    def transform_value(val, yes_pattern=yes_regex, no_pattern=no_regex):
        if pd.isnull(val) or val == '':
            return 0
        elif yes_pattern.search(str(val)):
            return 1
        elif no_pattern.search(str(val)):
            return 0
        else:
            return None

    # Aplicar las transformaciones a las columnas en columns_to_transform1
    for col in columns_to_transform1:
        df[col] = df[col].apply(lambda x: transform_value(x))

    # Aplicar las transformaciones a las columnas en columns_to_transform2
    for col in columns_to_transform2:
        df[col] = df[col].apply(lambda x: 1 if yes_regex.search(str(x)) else (0 if no_regex.search(str(x)) else None))

    logger.info("Valores binarios estandarizados!")

    return df

# 7. Imputación Nulos
def impute_missing_values_pd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa los valores faltantes en un DataFrame de Pandas basado en el tipo de cada columna.
    Para columnas de tipo string, se usa la moda; para columnas int64, se usa la mediana
    (redondeada si es necesario); y para columnas float64, se usa la mediana.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de Pandas sobre el cual se realizarán las imputaciones

    Returns
    -------
    pd.DataFrame
        DataFrame con los valores faltantes imputados.
    """
    logger.info("Iniciando la imputación de valores faltantes...")

    # Identificar columnas con valores nulos
    columns_with_nulls = [col for col in df.columns if df[col].isnull().sum() > 0]

    # Procesar cada columna con valores nulos
    for col in columns_with_nulls:
        col_type = df[col].dtype

        if pd.api.types.is_string_dtype(col_type):
            # Imputar con la moda para columnas de tipo string
            mode_value = df[col].mode().iloc[0]  # Obtener la moda
            df[col] = df[col].fillna(mode_value)
            logger.info(f"Imputados valores nulos en la columna '{col}' con la moda: {mode_value}")

        elif pd.api.types.is_integer_dtype(col_type):
            # Imputar con la mediana para columnas de tipo int64
            median_value = df[col].median()
            median_value = round(median_value)  # Redondear al entero más cercano si es necesario
            df[col] = df[col].fillna(median_value)
            logger.info(f"Imputados valores nulos en la columna '{col}' con la mediana redondeada: {median_value}")

        elif pd.api.types.is_float_dtype(col_type):
            # Imputar con la mediana para columnas de tipo float64
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            logger.info(f"Imputados valores nulos en la columna '{col}' con la mediana: {median_value}")

    logger.info("Imputación de valores faltantes completada!")

    return df
