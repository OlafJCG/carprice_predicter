# Libraries -----------------------------------------------------------------------------------
import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load Dataframe -----------------------------------------------------------------------------------
df = pd.read_parquet("files/datasets/intermediate/a02_preprocessing_rellena_elimina_anomalo.parquet")

# Seed -----------------------------------------------------------------------------------
seed = np.random.seed(42)

# Split Dataframe to train, valid and test sets -----------------------------------------------------------------------------------
X_train, X, y_train, y = train_test_split(df.drop('price', axis=1), df['price'], test_size=0.4, random_state=seed)
X_valid, X_test, y_valid, y_test = train_test_split(X, y, test_size=0.5, random_state=seed)

# Save Dataframe -----------------------------------------------------------------------------------
X_train.to_parquet("files/datasets/intermediate/a03_X_train.parquet")
X_valid.to_parquet("files/datasets/intermediate/a03_X_valid.parquet")
X_test.to_parquet("files/datasets/intermediate/a03_X_test.parquet")

y_train.to_frame().to_parquet("files/datasets/intermediate/a03_y_train.parquet")
y_valid.to_frame().to_parquet("files/datasets/intermediate/a03_y_valid.parquet")
y_test.to_frame().to_parquet("files/datasets/intermediate/a03_y_test.parquet")