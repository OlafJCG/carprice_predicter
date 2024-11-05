# Libraries -----------------------------------------------------------------------------------
import os
import sys
sys.path.append(os.getcwd())
import joblib

import pandas as pd
from sklearn.metrics import root_mean_squared_error
# Load DataFrame -----------------------------------------------------------------------------------
X_test = pd.read_parquet("files/datasets/intermediate/a04_X_test.parquet")
y_test = pd.read_parquet("files/datasets/intermediate/a04_y_test.parquet")

# Load Model -----------------------------------------------------------------------------------
model = joblib.load(
    "files/modeling_output/model_fit/b01_model_lgb.joblib"
)

# Test Model -----------------------------------------------------------------------------------
# Predice los precios de las observaciones de prueba
y_predict = model.predict(X_test)
# Muestra el rmse del conjunto de prueba
print("La recm del conjunto de prueba es:", root_mean_squared_error(y_test, y_predict))
