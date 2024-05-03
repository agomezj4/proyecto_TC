from typing import Dict, Any

import polars as pl
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 1. Validar tag de las fuentes
def validate_tags_pl(df: pl.DataFrame, tag_dict: pl.DataFrame) -> pl.DataFrame:
    """
    Valida que el número de tags identificados en source como raw sea igual al
    número de columnas en el dataframe utilizando Polars.

    Parameters
    ----------
    df : pl.DataFrame
        Dataframe que contiene los datos a validar
    tag_dict : pl.DataFrame
        Diccionario de etiquetas

    Returns
    -------
    pl.DataFrame or None
        Si las tags están validadas, retorna el DataFrame original. En otros casos,
        solo emite un log
    """

    # Registra un mensaje de información indicando el inicio del proceso de validación de tags
    logger.info("Iniciando la validación de tags...")

    # Calcula la cantidad de tags identificados como 'raw' en el tag dictionary
    len_raw = tag_dict.filter(pl.col("source") == "raw").height

    # Comprueba si el número de tags es mayor que el número de columnas en el DataFrame
    if len_raw != df.width:
        raise ValueError(
            "Tags faltantes en el dataframe" if len_raw > df.width else "Tags faltantes en el tag dictionary")

    # Si el número de tags es igual al número de columnas, emite un registro informativo y retorna el DataFrame 'df'
    logger.info("Tags validados!")

    return df



# 2. Validar tipos de datos
def validate_dtypes_pl(df: pl.DataFrame, tag_dict: pl.DataFrame) -> pl.DataFrame:
    """
    Revisa que la tipología de los datos en el dataframe sea la misma estipulada
    en el tag_dictionary.

    Parameters
    ----------
    df : pl.DataFrame
        Dataframe que se quiere validar
    tag_dict : pl.DataFrame
        Diccionario de etiquetas

    Returns
    -------
    pl.DataFrame or None
        Si las tipologías de datos están validadas, retorna el DataFrame original.
        En otros casos, solo emite un log
    """

    logger.info("Iniciando la validación de tipos de datos...")

    # Filtra las filas del tag dictionary donde "source" es "raw" y renombra la columna para el uso fácil
    tag_dict_raw = tag_dict.filter(pl.col("source") == "raw").with_columns(pl.col("tag").alias("column_name"))

    # Crear un diccionario de los tipos de datos esperados para cada columna, normalizando a minúsculas y eliminando espacios
    expected_types = {row['column_name']: row['data_type'].strip().lower() for row in tag_dict_raw.to_dicts()}

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
def change_names_pl(df: pl.DataFrame, tag_dict: pl.DataFrame) -> pl.DataFrame:
    """
    Cambia los nombres de las columnas de un DataFrame según un tag dictionary y devuelve
    un nuevo DataFrame con los nombres cambiados utilizando Polars.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame de Polars del cual se cambiarán los nombres de las columnas
    tag_dict: pl.DataFrame
        Diccionario de etiquetas

    Returns
    -------
    pl.DataFrame: Un nuevo DataFrame con las columnas renombradas según el tag dictionary
    """
    logger.info("Iniciando el cambio de nombres...")

    # Filtra las filas del tag dictionary donde "source" sea "raw" y selecciona las columnas "tag" y "name"
    tag_dict_filtered = tag_dict.filter(pl.col("source") == "raw").select(["tag", "name"])

    # Crea un diccionario de mapeo para cambiar los nombres de las columnas
    col_mapping = dict(zip(tag_dict_filtered['tag'].to_list(), tag_dict_filtered['name'].to_list()))

    # Cambia los nombres de las columnas según el tag dictionary "raw"
    df = df.rename(col_mapping)

    logger.info("Nombres de columnas cambiados!")

    return df


# 4. Cambiar tipos de datos
def change_dtype_pl(df: pl.DataFrame, tag_dict: pl.DataFrame) -> pl.DataFrame:
    """
    Cambia el tipo de datos de cada columna en un DataFrame al tipo de datos especificado
    en el tag dictionary usando Polars.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame de polars del cual se cambiarán los tipos de datos.
    tag_dict: polars.DataFrame
        Diccionario de etiquetas en un DataFrame de polars.

    Returns
    -------
    pl.DataFrame: DataFrame con las columnas cambiadas al nuevo tipo de datos.
    """
    logger.info("Iniciando el cambio de tipos de datos...")

    # Filtrar tag_dict para incluir solo las filas con "source" igual a "raw"
    tag_dict = tag_dict.filter(tag_dict["source"] == "raw")

    # Crear un diccionario de mapeo de tipos de datos
    type_mapping = dict(zip(tag_dict['name'].to_list(), tag_dict['data_type_new'].to_list()))

    # Preparar las columnas para la transformación
    transformed_columns = []
    for col in df.columns:
        if col in type_mapping:
            try:
                new_type = eval('pl.' + type_mapping[col]) if isinstance(type_mapping[col], str) else type_mapping[col]
                transformed_column = pl.col(col).cast(new_type).alias(col)
                transformed_columns.append(transformed_column)
            except Exception as e:
                print(f"Error al intentar castear la columna {col} a {type_mapping[col]}: {e}")

    # Aplicar las transformaciones si hay columnas a transformar
    if transformed_columns:
        df = df.with_columns(transformed_columns)

    logger.info("Dtypes cambiados!")

    return df


# 5. Eliminar acentos
def delete_accents_pl(df: pl.DataFrame) -> pl.DataFrame:
    """
    Elimina los acentos de las columnas identificadas como "str" de un DataFrame de Polars.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame de Polars del cual se eliminarán los acentos.

    Returns
    -------
    polars.DataFrame
        DataFrame de Polars con los acentos eliminados de las columnas especificadas.
    """
    logger.info("Iniciando la eliminación de acentos...")

    # Define una expresión regular que cubra los acentos y caracteres especiales más comunes
    accents_and_special_chars_pattern = r'[^\w\s]'  # Esto elimina todo lo que no sea alfanumérico o espacio

    # Seleccionar solo columnas de tipo string
    string_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]

    # Aplicar la eliminación solo a las columnas de tipo string
    update_expressions = [
        pl.col(col).str.replace_all(accents_and_special_chars_pattern, '').alias(col)
        for col in string_cols
    ]

    # Aplicar todas las expresiones de actualización al DataFrame manteniendo el resto de las columnas intactas
    df = df.with_columns(update_expressions)

    logger.info("Acentos eliminados!")

    return df

# 7. Re categorizar columnas
def standardize_binary_values_pl(
        df: pl.DataFrame,
        params: Dict[str, Any]
) -> pl.DataFrame:
    """
    Estandariza valores en columnas específicas de un DataFrame de Polars, transformando
    las diferentes formas de "sí" y "no" en valores binarios 1 y 0, respectivamente, y transformando
    los nulos o cadenas vacías a 0 en columnas específicas.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame de polars sobre el cual se realizarán las transformaciones.
    params: Dict[str, Any]
        Diccionario de parámetros processing.

    Returns
    -------
    pl.DataFrame: DataFrame con las columnas transformadas a valores binarios.
    """
    logger.info("Iniciando la estandarización de valores binarios...")

    # Columnas a transformar
    columns_to_transform1 = params['campos_binarios1']
    columns_to_transform2 = params['campos_binarios2']

    # Expresiones regulares comunes para "sí" y "no"
    yes_regex = r'\bs[ií]|\byes\b'  # Regex para capturar variaciones de "sí"
    no_regex = r'\bno\b'            # Regex para capturar variaciones de "no"

    # Lista de transformaciones para columns_to_transform1
    transformations_group1 = [
        pl.when(pl.col(col).is_null())
        .then(pl.lit(0).cast(pl.Int64))
        .when(pl.col(col).str.to_lowercase().str.contains(yes_regex))
        .then(pl.lit(1).cast(pl.Int64))
        .otherwise(
            pl.when(pl.col(col).str.to_lowercase().str.contains(no_regex))
            .then(pl.lit(0).cast(pl.Int64))
            .otherwise(pl.lit(None))
        )
        .alias(col)
        for col in columns_to_transform1
    ]

    # Lista de transformaciones para columns_to_transform2
    transformations_group2 = [
        pl.when(pl.col(col).str.to_lowercase().str.contains(yes_regex))
        .then(pl.lit(1).cast(pl.Int64))
        .when(pl.col(col).str.to_lowercase().str.contains(no_regex))
        .then(pl.lit(0).cast(pl.Int64))
        .otherwise(pl.lit(None))
        .alias(col)
        for col in columns_to_transform2
    ]

    # Aplicar todas las transformaciones en una sola operación
    df = df.with_columns(transformations_group1 + transformations_group2)

    logger.info("Valores binarios estandarizados!")

    return df

# 7. Imputación Nulos
def impute_missing_values_pl(df: pl.DataFrame) -> pl.DataFrame:
    """
    Imputa los valores faltantes en un DataFrame de Polars basado en el tipo de cada columna.
    Para columnas de tipo string, se usa la moda; para columnas int64, se usa la mediana
    (redondeada si es necesario); y para columnas float64, se usa la mediana.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame de Polars sobre el cual se realizarán las imputaciones.

    Returns
    -------
    pl.DataFrame
        DataFrame con los valores faltantes imputados.
    """
    logger.info("Iniciando la imputación de valores faltantes...")

    # Identificar columnas con valores nulos
    columns_with_nulls = [col for col in df.columns if df[col].is_null().sum() > 0]

    # Procesar cada columna con valores nulos
    for col in columns_with_nulls:
        col_type = df[col].dtype

        if col_type == pl.Utf8:
            # Imputar con la moda para columnas de tipo string
            mode_value = df[col].mode().to_list()[0]  # Obtener la moda como un valor en una lista
            df = df.with_columns(pl.col(col).fill_null(mode_value).alias(col))
            logger.info(f"Imputados valores nulos en la columna '{col}' con la moda: {mode_value}")

        elif col_type == pl.Int64:
            # Imputar con la mediana para columnas de tipo int64
            median_value = df[col].median()
            median_value = round(median_value)  # Redondear al entero más cercano si es necesario
            df = df.with_columns(pl.col(col).fill_null(median_value).alias(col))
            logger.info(f"Imputados valores nulos en la columna '{col}' con la mediana redondeada: {median_value}")

        elif col_type == pl.Float64:
            # Imputar con la mediana para columnas de tipo float64
            median_value = df[col].median()
            df = df.with_columns(pl.col(col).fill_null(median_value).alias(col))
            logger.info(f"Imputados valores nulos en la columna '{col}' con la mediana: {median_value}")

    logger.info("Imputación de valores faltantes completada!")

    return df
