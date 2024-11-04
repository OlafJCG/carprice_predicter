# Libraries -----------------------------------------------------------------------------------
import pandas as pd
pd.options.display.max_columns = None

# Load Dataset -----------------------------------------------------------------------------------
df = pd.read_csv("files/datasets/input/autos.csv")

# Muestra la información general del conjunto de datos
df.info()

# Imprime una muestra de los datos
df.sample(5)

# Revisa la distribución de columnas con datos numéricos.
df.describe()
