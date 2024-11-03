# Libraries -----------------------------------------------------------------------------------
import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OrdinalEncoder, RobustScaler


# Load Dataframe -----------------------------------------------------------------------------------
X_train = pd.read_parquet("files/datasets/intermediate/a03_X_train.parquet")
X_valid = pd.read_parquet("files/datasets/intermediate/a03_X_valid.parquet")
X_test = pd.read_parquet("files/datasets/intermediate/a03_X_test.parquet")

y_train = pd.read_parquet("files/datasets/intermediate/a03_y_train.parquet")
y_valid = pd.read_parquet("files/datasets/intermediate/a03_y_valid.parquet")
y_test = pd.read_parquet("files/datasets/intermediate/a03_y_test.parquet")

# Encoder -----------------------------------------------------------------------------------
# Crea una instancia para el codificador para la columna 'repaired' con el orden de las etiquetas y aplica fit con la columna 'repaired' del conjunto de entrenamiento.
lr_ordinal_encoder = OrdinalEncoder(categories=[["unknown", "yes", "no"]]).fit(X_train[['repaired']])
# Transforma las etiquetas del conjunto de entrenamiento
X_train[['repaired']] = lr_ordinal_encoder.transform(X_train[['repaired']])
X_train[['repaired']] = X_train[['repaired']].astype('int64')
# Transforma las etiquetas del conjunto de validación y prueba
X_valid[['repaired']] = lr_ordinal_encoder.transform(X_valid[['repaired']])
X_valid[['repaired']] = X_valid[['repaired']].astype('int64')
X_test[['repaired']] = lr_ordinal_encoder.transform(X_test[['repaired']])
X_test[['repaired']] = X_test[['repaired']].astype('int64')
# Codifica algunas caracteristicas nominales con BinaryEncoder.
cols_to_binary_encoder = ['vehicle_type', 'gearbox', 'model', 'fuel_type', 'brand']
binary_encoder = BinaryEncoder(cols=cols_to_binary_encoder).fit(X_train[cols_to_binary_encoder])
# Crea una variable con las nuevas columnas
X_train_bin_cols = binary_encoder.transform(X_train[cols_to_binary_encoder])
X_valid_bin_cols = binary_encoder.transform(X_valid[cols_to_binary_encoder])
X_test_bin_cols = binary_encoder.transform(X_test[cols_to_binary_encoder])
# Elimina las columnas que vamos a reemplazar
X_train = X_train.drop(cols_to_binary_encoder, axis=1)
X_valid = X_valid.drop(cols_to_binary_encoder, axis=1)
X_test = X_test.drop(cols_to_binary_encoder, axis=1)
# Concatena los dataframes que reemplazan las columnas codificadas
X_train = pd.concat([X_train, X_train_bin_cols], axis=1)
X_valid = pd.concat([X_valid, X_valid_bin_cols], axis=1)
X_test = pd.concat([X_test, X_test_bin_cols], axis=1)

# Scaler -----------------------------------------------------------------------------------
# Crea un transformador para las características numéricas a escalar
numeric_cols_for_scaler = ['registration_year', 'power', 'kilometer']
rob_scaler = RobustScaler().fit(X_train[numeric_cols_for_scaler])
# Escala las características numéricas con RobustScaler para que la normalización sea robusta a los datos anomalos
X_train[numeric_cols_for_scaler] = rob_scaler.transform(X_train[numeric_cols_for_scaler])
X_valid[numeric_cols_for_scaler] = rob_scaler.transform(X_valid[numeric_cols_for_scaler])
X_test[numeric_cols_for_scaler] = rob_scaler.transform(X_test[numeric_cols_for_scaler])

# Save Dataframe -----------------------------------------------------------------------------------
X_train.to_parquet("files/datasets/intermediate/a04_X_train.parquet")
X_valid.to_parquet("files/datasets/intermediate/a04_X_valid.parquet")
X_test.to_parquet("files/datasets/intermediate/a04_X_test.parquet")

y_train.to_parquet("files/datasets/intermediate/a04_y_train.parquet")
y_valid.to_parquet("files/datasets/intermediate/a04_y_valid.parquet")
y_test.to_parquet("files/datasets/intermediate/a04_y_test.parquet")