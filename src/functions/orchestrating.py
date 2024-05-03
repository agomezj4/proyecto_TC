import polars as pl
import os
import sys
import yaml
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ajustar el path del sistema para incluir el directorio 'src' al nivel superior
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from functions.processing import (validate_tags_pl,
                                  validate_dtypes_pl,
                                  change_names_pl,
                                  change_dtype_pl,
                                  delete_accents_pl,
                                  standardize_binary_values_pl,
                                  impute_missing_values_pl)


# Directorios para los archivos de parámetros y los datos
parameters_directory = os.path.join(project_root, 'src', 'parameters')
data_raw_directory = os.path.join(project_root, 'data', 'raw')
data_processed_directory = os.path.join(project_root, 'data', 'processed')
data_features_directory = os.path.join(project_root, 'data', 'features')


# Lista todos los archivos YAML en el directorio especificado
yaml_files = [f for f in os.listdir(parameters_directory) if f.endswith('.yml')]

# Diccionario para guardar los parámetros cargados
parameters = {}

# Carga cada archivo YAML
for yaml_file in yaml_files:
    with open(os.path.join(parameters_directory, yaml_file), 'r') as file:
        data = yaml.safe_load(file)
        key_name = f'parameters_{yaml_file.replace(".yml", "")}'
        parameters[key_name] = data

def run_processing():

    logger.info("INICIO ETAPA PROCESAMIENTO DE DATOS...")

    # Cargar datos
    tag_dict_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['tag_dict_path'])
    raw_data_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['raw_data_path'])

    tag_dict = pl.read_excel(tag_dict_path)
    data_raw = pl.read_csv(raw_data_path)
    logger.info("Tag dict y datos raw cargados correctamente...")

    # Validar etiquetas
    data_validate_tags = validate_tags_pl(data_raw, tag_dict)

    # Validar tipos de datos
    data_validate_dtypes = validate_dtypes_pl(data_validate_tags, tag_dict)

    # Cambio de nombres de columnas
    data_change_names = change_names_pl(data_validate_dtypes, tag_dict)

    # Cambio de tipos de datos
    data_change_dtype = change_dtype_pl(data_change_names, tag_dict)

    # Eliminar acentos
    data_delete_accents = delete_accents_pl(data_change_dtype)

    # Estandarizar a valores binarios
    data_standardize_binary_values = standardize_binary_values_pl(data_delete_accents, parameters['parameters_processing'])

    # Imputar valores faltantes
    data_processing = impute_missing_values_pl(data_standardize_binary_values)

    # Guardar datos procesados
    processed_data_path = os.path.join(data_processed_directory, parameters['parameters_catalog']['processed_data_path'])
    data_processing.write_csv(processed_data_path)
    logger.info("Datos procesados guardados correctamente!")

    logger.info("FIN ETAPA PROCESAMIENTO DE DATOS!")
