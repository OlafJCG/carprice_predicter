# Libraries -----------------------------------------------------------------------------------
import os
import sys
sys.path.append(os.getcwd())
import numpy as np

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


# Load DataFrames -----------------------------------------------------------------------------------
X_train = pd.read_parquet("files/datasets/intermediate/a04_X_train.parquet")
X_valid = pd.read_parquet("files/datasets/intermediate/a04_X_valid.parquet")
X_test = pd.read_parquet("files/datasets/intermediate/a04_X_test.parquet")

y_train = pd.read_parquet("files/datasets/intermediate/a04_y_train.parquet")
y_train = pd.Series(y_train['price'])
y_valid = pd.read_parquet("files/datasets/intermediate/a04_y_valid.parquet")
y_valid = pd.Series(y_valid['price'])
y_test = pd.read_parquet("files/datasets/intermediate/a04_y_test.parquet")
y_test = pd.Series(y_test['price'])

# Seed -----------------------------------------------------------------------------------
seed = np.random.seed(42)

# Model Training -----------------------------------------------------------------------------------
# Regresión Lineal 
# Crea una instancia del modelo
clf_lr = LinearRegression()
# Entrena el modelo
clf_lr.fit(X_train, y_train)

# XGBoost
# Entrena el modelo con los mejores parámetros
clf_xgb = XGBRegressor(learning_rate=0.07, max_depth=7).fit(X_train, y_train)

# Bosque Aleatorio.
# Entrena el modelo con los mejores hiperparámetros
clf_rfr = RandomForestRegressor(criterion='poisson', max_depth=43, random_state=seed).fit(X_train, y_train)

# CatBoostClassifier
# Entrena el modelo con los mejores parámetros
cat_model = CatBoostRegressor(loss_function='RMSE', learning_rate=0.5, iterations=150, random_state=seed).fit(X_train, y_train)

# LightGBM
# Entrena el modelo con los mejores parámetros
lgb_model = LGBMRegressor(metric='rmse', boosting_type='gbdt', learning_rate=0.5, num_leaves=31, random_state=seed).fit(X_train, y_train)

# Save Model -----------------------------------------------------------------------------------
joblib.dump(
    clf_lr,
    f"files/modeling_output/model_fit/b01_model_lr.joblib"
)

joblib.dump(
    clf_xgb,
    f"files/modeling_output/model_fit/b01_model_xgb.joblib"
)

joblib.dump(
    clf_rfr,
    f"files/modeling_output/model_fit/b01_model_rfr.joblib"
)

joblib.dump(
    cat_model, 
    f"files/modeling_output/model_fit/b01_model_cat.joblib"
)

joblib.dump(
    lgb_model,
    f"files/modeling_output/model_fit/b01_model_lgb.joblib"
)