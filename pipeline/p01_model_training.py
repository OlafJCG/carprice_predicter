# Libraries -----------------------------------------------------------------------------------
import os
import sys
sys.path.append(os.getcwd())

import params as params

# Defining executable file extensions -----------------------------------------------------------------------------------
if params.operating_system == "Windows":
    binary_extensions = ".exe"
else:
    binary_extensions = ""

# Info -----------------------------------------------------------------------------------
print("-----------------------------------------------------------\nPreprocesamiento...\n-----------------------------------------------------------")

# Preprocessing -----------------------------------------------------------------------------------
os.system(f"python{binary_extensions} preprocessing/a01_preprocessing_elimina_cols_y_nombres_estandar.py")
os.system(f"python{binary_extensions} preprocessing/a02_preprocessing_rellena_elimina_anomalo.py")
os.system(f"python{binary_extensions} preprocessing/a03_split.py")
os.system(f"python{binary_extensions} preprocessing/a04_scaler_encoder.py")

# Info -----------------------------------------------------------------------------------
print("-----------------------------------------------------------\nEntrenando Modelos...\n-----------------------------------------------------------")

# Models Training -----------------------------------------------------------------------------------
os.system(f"python{binary_extensions} models/b01_models_creation.py")

# Info -----------------------------------------------------------------------------------

# Info -----------------------------------------------------------------------------------
print("-----------------------------------------------------------\nEntrenamiento finanizado...\n-----------------------------------------------------------")