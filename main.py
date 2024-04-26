# Librearias
import polars as pl
import sys


########## Processing Data ##########
sys.path.append('src/')
from paths.path import raw_data_path, tag_dict_path
from functions.process import validate_tags_pl, validate_dtypes_pl, change_names_pl


# Load data
tag_dict = pl.read_excel(tag_dict_path)
data = pl.read_csv(raw_data_path)

# Validar campos de las fuentes
data = validate_tags_pl(data, tag_dict)

# Validar tipos de datos
data = validate_dtypes_pl(data, tag_dict)


