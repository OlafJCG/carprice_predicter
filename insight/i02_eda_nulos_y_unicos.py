# Libraries -----------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# Load Dataset -----------------------------------------------------------------------------------
df = pd.read_parquet("files/datasets/intermediate/a01_preprocessing_elimina_cols_y_nombres_estandar.parquet")

# EDA -----------------------------------------------------------------------------------
# Crea variables que guarden los nombres de las columnas en español para la visualización
nombres_columnas = ['Precio','Tipo de vehículo', 'Año de matriculación', 'Caja de cambios', 'Potencia (CV)', 'Modelo', 'Kilometraje', 'Tipo de combustible', 'Marca', 'Reparado', 'Código postal']
numeric_cols_names = ['price', 'registration_year', 'power', 'kilometer', 'number_of_pictures', 'postal_code']
numeric_cols_index = [0,2,4,6,7,11]
categorical_cols_index = [1,3,5,7,9,10,11]
# Revisa los nombres únicos de las columnas de tipo de vehiculo y tipo de caja de cambios.
for col_name in df[['vehicle_type', 'gearbox', 'fuel_type', 'brand', 'repaired']]:
    print(f'{col_name}:\n',df[col_name].sort_values().unique())
# Revisa el porcentaje de valores nulos en cada columna.
{col: [print(f"Para la columna {col}: \nDatos nulos: {df[col].isnull().sum()}, Porcentaje: {np.round(np.mean(df[col].isnull()*100), 3)}%")] for col in df.columns if df[col].isnull().any()}
# Muestra el porcentaje de valores nulos en todo el dataset.
print(f'Total de valores nulos: {np.round(np.absolute(df.dropna().count() / df.shape[0] -1)[0] *100,3)}%')
# Muestra un gráfico que muestre el porcentaje de nulos en cada columna (Descomenta para mostrar el gráfico)
# sns.displot(
#     data=df.isnull().melt(value_name='nulos'),
#     y='variable',
#     hue='nulos',
#     multiple='fill'
# )
# Quita el comentario de la siguiente linea para mostrar el gráfico
# plt.show()