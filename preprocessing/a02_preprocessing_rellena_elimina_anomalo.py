# Libraries -----------------------------------------------------------------------------------
import os
import sys
sys.path.append(os.getcwd())

import pandas as pd

# Load Dataset -----------------------------------------------------------------------------------
df = pd.read_parquet("files/datasets/intermediate/a01_preprocessing_elimina_cols_y_nombres_estandar.parquet")

# Clean Data -----------------------------------------------------------------------------------
# Rellena los datos nulos de la columna "repaired" con "unknown"
df['repaired'] = df['repaired'].fillna('unknown')
# Rellena los datos de "price" por debajo del primer cuartil con la media.
df.loc[df['price']<df['price'].quantile(0.25), 'price'] = df['price'].mean()
# Rellena los datos de "power" por debajo del primer cuartil con la media.
df.loc[df['power']<df['power'].quantile(0.25), 'power'] = df['power'].mean()
# Elimina los datos anomalos en la columna "registration_year"
df = df.query("registration_year > 1886 and registration_year < 2024")
df = df.query("power > power.quantile(0.25) and power < 1000")
df = df.dropna().reset_index(drop=True)

# Save Dataset -----------------------------------------------------------------------------------
df.to_parquet("files/datasets/intermediate/a02_preprocessing_rellena_elimina_anomalo.parquet")