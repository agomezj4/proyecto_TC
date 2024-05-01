from typing import Dict, List, Any

import polars as pl
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Guardar datos en una ruta especifica
def save_csv_pl(df: pl.DataFrame, params: Dict[str, Any]) -> None:
    """
    Guarda un DataFrame de Polars en una ruta específica como un archivo CSV, utilizando parámetros dados.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame de Polars que será guardado en un archivo CSV.
    params : Dict[str, Any]
        Diccionario de parámetros que incluye la ruta de guardado bajo la clave 'file_path'.

    Returns
    -------
    None
    """
    logger.info("Iniciando el proceso de guardado del DataFrame como CSV...")

    try:
        # Extraer la ruta de archivo de los parámetros
        file_path = params.get('file_path')
        if not file_path:
            logger.error("No se proporcionó una ruta de archivo válida en los parámetros.")
            raise ValueError("La clave 'file_path' no está presente en los parámetros o está vacía.")

        # Guardar el DataFrame en la ruta especificada
        df.write_csv(file_path)
        logger.info(f"DataFrame guardado exitosamente en {file_path}")

    except Exception as e:
        logger.error(f"Error al guardar el DataFrame: {e}")
        raise


# Cargar datos de una ruta especifica