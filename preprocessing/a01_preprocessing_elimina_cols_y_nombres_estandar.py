# Libraries -----------------------------------------------------------------------------------
import os
import sys
sys.path.append(os.getcwd())

import pandas as pd

from functions import text_changer

# Load Dataset -----------------------------------------------------------------------------------
df = pd.read_csv("files/datasets/input/autos.csv")

# Fixing Data -----------------------------------------------------------------------------------
# Elimina las columnas 0, 8, 12, 13, 14, 15 son fechas y características que no son necesarias para nuestra tarea.
df = df.drop(df.columns[[0,8,12,13,14,15]], axis=1)
# Cambia los nombres de tipo camel o snake y lower
df.columns = pd.Series(df.columns).apply(text_changer.split_camel_to_snake)
df.rename(columns={'not_repaired':'repaired'},inplace=True)

# Save Dataset -----------------------------------------------------------------------------------
df.to_parquet("files/datasets/intermediate/a01_preprocessing_elimina_cols_y_nombres_estandar.parquet")