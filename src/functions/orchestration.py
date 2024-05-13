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

from functions.featuring import (new_features_pl,
                                 add_target_variable_pl,
                                 one_hot_encoding_pl,
                                 random_forest_selection_pl,
                                 conditional_entropy_selection_pl,
                                 intersect_top_features_pl)

from functions.model_input import (min_max_scaler_pl,
                                   balance_target_variable_pl,
                                   train_test_split_pl)

from functions.models import (a_pl,
                              b_pl,
                              c_pl)


# Directorios para los archivos de parámetros y los datos
parameters_directory = os.path.join(project_root, 'src', 'parameters')
data_raw_directory = os.path.join(project_root, 'data', 'raw')
data_processed_directory = os.path.join(project_root, 'data', 'processed')
target_directory = os.path.join(project_root, 'data', 'processed')
data_features_directory = os.path.join(project_root, 'data', 'featured')
data_train_directory = os.path.join(project_root, 'data', 'model_input', 'train')
data_test_directory = os.path.join(project_root, 'data', 'model_input', 'test')
data_model_directory = os.path.join(project_root, 'data', 'models')


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

    # Cargar datos
    tag_dict_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['tag_dict_path'])
    raw_data_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['raw_data_path'])

    tag_dict = pl.read_excel(tag_dict_path)
    data_raw = pl.read_csv(raw_data_path)

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

def run_featuring():
    # Cargar datos procesados
    processed_data_path = os.path.join(data_processed_directory, parameters['parameters_catalog']['processed_data_path'])
    data_processed = pl.read_csv(processed_data_path)

    # Generar características
    data_features = new_features_pl(data_processed, parameters['parameters_featuring'])

    # Añadir target al conjunto de datos
    target_path = os.path.join(target_directory, parameters['parameters_catalog']['target_column_path'])
    target = pl.read_csv(target_path)
    add_target_variable = add_target_variable_pl(data_features, target, parameters['parameters_featuring'])

    # Codificación one-hot
    data_one_hot = one_hot_encoding_pl(add_target_variable, parameters['parameters_featuring'])

    # Selección de características con Random Forest
    data_random_forest = random_forest_selection_pl(data_one_hot, parameters['parameters_featuring'])

    # Selección de características con Entropía Condicional
    data_conditional_entropy = conditional_entropy_selection_pl(data_one_hot, parameters['parameters_featuring'])

    # Intersección de las mejores características
    data_features = intersect_top_features_pl(data_random_forest, data_conditional_entropy, data_one_hot, parameters['parameters_featuring'])

    # Guardar datos de características
    features_data_path = os.path.join(data_features_directory, parameters['parameters_catalog']['features_data_path'])
    data_features.write_csv(features_data_path)

def run_model_input():
    # Cargar datos de características
    features_data_path = os.path.join(data_features_directory, parameters['parameters_catalog']['features_data_path'])
    data_features = pl.read_csv(features_data_path)

    # Escalar características
    data_scaled = min_max_scaler_pl(data_features)

    # Balancear la variable target
    data_balanced = balance_target_variable_pl(data_scaled, parameters['parameters_model_input'])

    # Separar datos en entrenamiento y prueba
    train_data, test_data = train_test_split_pl(data_balanced, parameters['parameters_model_input'])

    # Guardar datos de entrenamiento y prueba
    train_data_path = os.path.join(data_train_directory, parameters['parameters_catalog']['train_data_path'])
    test_data_path = os.path.join(data_test_directory, parameters['parameters_catalog']['test_data_path'])
    train_data.write_csv(train_data_path)
    test_data.write_csv(test_data_path)

def run_models():
    # Cargar datos de entrenamiento
    train_data_path = os.path.join(data_train_directory, parameters['parameters_catalog']['train_data_path'])
    train_data = pl.read_csv(train_data_path)

    # Entrenar modelos
    model_a = a_pl(train_data, parameters['parameters_models'])
    model_b = b_pl(train_data, parameters['parameters_models'])
    model_c = c_pl(train_data, parameters['parameters_models'])

    # Guardar modelos
    model_a_path = os.path.join(data_model_directory, parameters['parameters_catalog']['model_a_path'])
    model_b_path = os.path.join(data_model_directory, parameters['parameters_catalog']['model_b_path'])
    model_c_path = os.path.join(data_model_directory, parameters['parameters_catalog']['model_c_path'])
    model_a.write_csv(model_a_path)
    model_b.write_csv(model_b_path)
    model_c.write_csv(model_c_path)
