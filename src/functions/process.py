from typing import Dict, List, Any

import polars as pl
import unicodedata
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

    logger.info("Nombres de columnas cambiados")

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


# 7. Nulos
