import pandas as pd
import os
import sys
import yaml
import pickle
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Ajustar el path del sistema para incluir el directorio 'src' al nivel superior
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


# Directorios para los archivos de parámetros y los datos
parameters_directory = os.path.join(project_root, 'src', 'parameters')
data_raw_directory = os.path.join(project_root, 'data', 'raw')
data_processed_directory = os.path.join(project_root, 'data', 'processed')
target_directory = os.path.join(project_root, 'data', 'processed')
data_features_directory = os.path.join(project_root, 'data', 'featured')
data_train_directory = os.path.join(project_root, 'data', 'model_input', 'train')
data_test_directory = os.path.join(project_root, 'data', 'model_input', 'test')
data_validation_directory = os.path.join(project_root, 'data', 'model_input', 'validation')
data_model_basic_directory = os.path.join(project_root, 'data', 'models', 'basic')
data_model_ensemble_directory = os.path.join(project_root, 'data', 'models', 'ensemble')
data_model_selection_directory = os.path.join(project_root, 'data', 'models', 'model_selection')


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


# Importar funciones de los pipelines

# Nodos pipeline de procesamiento
from functions.processing import (validate_tags_pd,
                                  validate_dtypes_pd,
                                  change_names_pd,
                                  change_dtype_pd,
                                  delete_accents_pd,
                                  standardize_binary_values_pd,
                                  impute_missing_values_pd)

# Pipeline de procesamiento
def run_processing():

    # Cargar datos
    tag_dict_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['tag_dict_path'])
    raw_data_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['raw_data_path'])

    tag_dict = pd.read_excel(tag_dict_path)
    data_raw = pd.read_csv(raw_data_path)

    # Validar etiquetas
    data_validate_tags = validate_tags_pd(data_raw, tag_dict)

    # Validar tipos de datos
    data_validate_dtypes = validate_dtypes_pd(data_validate_tags, tag_dict)

    # Cambio de nombres de columnas
    data_change_names = change_names_pd(data_validate_dtypes, tag_dict)

    # Cambio de tipos de datos
    data_change_dtype = change_dtype_pd(data_change_names, tag_dict)

    # Eliminar acentos
    data_delete_accents = delete_accents_pd(data_change_dtype)

    # Estandarizar a valores binarios
    data_standardize_binary_values = standardize_binary_values_pd(data_delete_accents, parameters['parameters_processing'])

    # Imputar valores faltantes
    data_processing = impute_missing_values_pd(data_standardize_binary_values)

    # Guardar datos procesados
    processed_data_path = os.path.join(data_processed_directory, parameters['parameters_catalog']['processed_data_path'])
    data_processing.to_parquet(processed_data_path)


# Nodos pipeline de ingeniería de características
from functions.featuring import (new_features_pd,
                                 add_target_variable_pd,
                                 one_hot_encoding_pd,
                                 random_forest_selection_pd,
                                 conditional_entropy_selection_pd,
                                 intersect_top_features_pd)

# Pipeline de ingeniería de características
def run_featuring():
    # Cargar datos procesados
    processed_data_path = os.path.join(data_processed_directory, parameters['parameters_catalog']['processed_data_path'])
    data_processed = pd.read_parquet(processed_data_path)

    # Generar características
    data_features = new_features_pd(data_processed, parameters['parameters_featuring'])

    # Añadir target al conjunto de datos
    target_path = os.path.join(target_directory, parameters['parameters_catalog']['target_column_path'])
    target = pd.read_csv(target_path)
    add_target_variable = add_target_variable_pd(data_features, target, parameters['parameters_featuring'])

    # Codificación one-hot
    data_one_hot = one_hot_encoding_pd(add_target_variable, parameters['parameters_featuring'])

    # Selección de características con Random Forest
    data_random_forest = random_forest_selection_pd(data_one_hot, parameters['parameters_featuring'])

    # Selección de características con Entropía Condicional
    data_conditional_entropy = conditional_entropy_selection_pd(data_one_hot, parameters['parameters_featuring'])

    # Intersección de las mejores características
    data_features = intersect_top_features_pd(data_random_forest, data_conditional_entropy, data_one_hot, parameters['parameters_featuring'])

    # Guardar datos de características
    features_data_path = os.path.join(data_features_directory, parameters['parameters_catalog']['features_data_path'])
    data_features.to_parquet(features_data_path)


# Nodos pipeline model input
from functions.model_input import (min_max_scaler_pd,
                                   balance_target_variable_pd,
                                   train_test_split_pd)

# Pipeline model input
def run_model_input():
    # Cargar datos de características
    features_data_path = os.path.join(data_features_directory, parameters['parameters_catalog']['features_data_path'])
    data_features = pd.read_parquet(features_data_path)

    # Escalar características
    data_scaled = min_max_scaler_pd(data_features)

    # Balancear la variable target
    data_balanced = balance_target_variable_pd(data_scaled, parameters['parameters_model_input'])

    # Separar datos en entrenamiento y prueba
    train_data, test_data, validation_data = train_test_split_pd(data_balanced, parameters['parameters_model_input'])

    # Guardar datos de entrenamiento y prueba
    train_data_path = os.path.join(data_train_directory, parameters['parameters_catalog']['train_data_path'])
    test_data_path = os.path.join(data_test_directory, parameters['parameters_catalog']['test_data_path'])
    validation_data_path = os.path.join(data_validation_directory,
                                        parameters['parameters_catalog']['validation_data_path'])
    train_data.to_parquet(train_data_path)
    test_data.to_parquet(test_data_path)
    validation_data.to_parquet(validation_data_path)


# Nodos pipeline de entrenamiento de modelos
from functions.models import (train_models_pd)

# Pipeline de entrenamiento de modelos
def run_models():
    # Cargar datos de entrenamiento
    train_data_path = os.path.join(data_train_directory, parameters['parameters_catalog']['train_data_path'])
    train_data = pd.read_parquet(train_data_path)

    # Entrenar modelos
    best_models = train_models_pd(train_data, parameters['parameters_models'])

    # Guardar modelos
    basic_model_path = os.path.join(data_model_basic_directory, parameters['parameters_catalog']['basic_model_path'])
    ensemble_model_path = os.path.join(data_model_ensemble_directory, parameters['parameters_catalog']['ensemble_model_path'])

    with open(basic_model_path, 'wb') as f:
        f.write(best_models['basic']['pickle'])
    with open(ensemble_model_path, 'wb') as f:
        f.write(best_models['ensemble']['pickle'])


# Nodos pipeline de selección de modelos
from functions.model_selection import (optimize_train_knn,
                                       optimize_train_xgboost,
                                       ab_test_models)

# Pipeline de selección de modelos
def run_model_selection():
    # Cargar datos de validación
    val_data_path = os.path.join(data_validation_directory, parameters['parameters_catalog']['validation_data_path'])
    val_data = pd.read_parquet(val_data_path)

    # Cargar datos de prueba
    test_data_path = os.path.join(data_test_directory, parameters['parameters_catalog']['test_data_path'])
    test_data = pd.read_parquet(test_data_path)

    # Cargar modelos
    basic_model_path = os.path.join(data_model_basic_directory, parameters['parameters_catalog']['basic_model_path'])
    ensemble_model_path = os.path.join(data_model_ensemble_directory, parameters['parameters_catalog']['ensemble_model_path'])
    with open(basic_model_path, 'rb') as f:
        basic_model = pickle.load(f)
    with open(ensemble_model_path, 'rb') as f:
        ensemble_model = pickle.load(f)

    # Optimizar hiperparámetros de KNN
    best_knn = optimize_train_knn(basic_model, val_data, test_data, parameters['parameters_model_selection'])

    # Optimizar hiperparámetros de XGBoost
    best_xgboost = optimize_train_xgboost(ensemble_model, val_data, test_data, parameters['parameters_model_selection'])

    # A/B testing
    ab_model_selection = ab_test_models(best_knn, best_xgboost, test_data, parameters['parameters_model_selection'])

    # Guardar modelos seleccionados
    best_basic_model_path = os.path.join(data_model_selection_directory, parameters['parameters_catalog']['best_basic_model_path'])
    best_ensemble_model_path = os.path.join(data_model_selection_directory, parameters['parameters_catalog']['best_ensemble_model_path'])
    best_model_ab_test_path = os.path.join(data_model_selection_directory, parameters['parameters_catalog']['best_model_ab_test_path'])

    with open(best_basic_model_path, 'wb') as f:
        pickle.dump(best_knn, f)
    with open(best_ensemble_model_path, 'wb') as f:
        pickle.dump(best_xgboost, f)
    with open(best_model_ab_test_path, 'wb') as f:
        pickle.dump(ab_model_selection, f)
