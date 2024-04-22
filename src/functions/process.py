from typing import Dict, List, Any, Tuple

import pandas as pd
import unicodedata
import logging

# 1. Validar tipos de datos
# 2. Re- nombrar
# 3. Eliminar acentos
# 4. Duplicaods
# 5. Nulos
def delete_accents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina los acentos de las columnas identificadas como "object" de un DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de pandas del cual se eliminarán los acentos.

    Returns
    -------
    pandas.DataFrame
        DataFrame de pandas con los acentos eliminados de las columnas especificadas.
    """
    logging.info("Eliminando acentos de las columnas de tipo 'object'")

    # Lista de nombres de columnas en las que se eliminarán los acentos
    columns = df.select_dtypes(include=["object"]).columns.tolist()

    # Define una función para eliminar acentos y caracteres especiales
    def remove_accents(x):
        return unicodedata.normalize("NFD", str(x)).encode("ascii", "ignore").decode("utf-8")

    # Aplica la función remove_accents a cada columna especificada
    for col in columns:
        df[col] = df[col].map(remove_accents)

    logging.info("Acentos eliminados")

    return df
