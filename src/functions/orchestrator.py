import polars as pl
import sys
import yaml

sys.path.append('src/')
from paths.path import raw_data_path, parameters_processing_path
from functions.processing import validate_tags_pl, validate_dtypes_pl

yaml_files = [
    ('parameters_processing', parameters_processing_path),
]

parameters = {}

for identifier, yaml_path in yaml_files:
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        key_name = f'parameters_{identifier}'
        parameters[key_name] = data

def run_processing():
    processing_data()
    # Cargar datos
    data = pl.read_csv(raw_data_path)

    # Validar etiquetas
    data = validate_tags_pl(data)

    # Validar tipos de datos
    data = validate_dtypes_pl(data)

    # Limpieza adicional de datos
    data = clean_data(data)

    # Guardar datos procesados
    save_processed_data(data, processed_data_path)

def run_feature_engineering():
    generate_features()

def train_model():
    train_model()
