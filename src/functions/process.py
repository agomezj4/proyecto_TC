from typing import Dict, List, Any, Tuple

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

    # Registra un mensaje de información indicando el inicio del proceso de cambio de nombres
    logger.info("Iniciando el cambio de nombres...")

    # Filtra las filas del tag dictionary donde "source" sea "raw" y crea una nueva columna 'column_name'
    tag_dict_filtered = tag_dict.filter(pl.col("source") == "raw").select(["tag", "name"])

    # Crea un diccionario de mapeo para cambiar los nombres de las columnas
    col_mapping = {row['tag']: row['name'] for row in tag_dict_filtered.collect()}

    # Cambia los nombres de las columnas según el tag dictionary "raw"
    df = df.rename(col_mapping)

    # Registra un mensaje informativo
    logger.info("Nombres de columnas cambiados")

    return df



# 4. Cambiar tipos de datos
# 5. Eliminar acentos
# 6. Duplicaods
# 7. Nulos
#8. Eliminar Acentos
