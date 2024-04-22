import pandas as pd
import sys


########## Process Data ##########
sys.path.append('src/')
from paths.path import raw_data_path
from functions.process import delete_accents


# Load data
data = pd.read_csv(raw_data_path)
data.info()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
data.info()

# Delete accents
data = delete_accents(data)


