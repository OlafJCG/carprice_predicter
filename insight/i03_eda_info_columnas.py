# Libraries -----------------------------------------------------------------------------------
import os
import sys 
sys.path.append(os.getcwd())
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt

from functions.plot_loop_crafter import traza_plots_condicionados
from functions.boxplot_calculator import iqr_calc

# Load Dataframe -----------------------------------------------------------------------------------
df = pd.read_parquet("files/datasets/intermediate/a01_preprocessing_elimina_cols_y_nombres_estandar.parquet")

# EDA -----------------------------------------------------------------------------------
# Crea variables que guarden los nombres de las columnas en español para la visualización
nombres_columnas = ['Precio','Tipo de vehículo', 'Año de matriculación', 'Caja de cambios', 'Potencia (CV)', 'Modelo', 'Kilometraje', 'Tipo de combustible', 'Marca', 'Reparado', 'Código postal']
numeric_cols_names = ['price', 'registration_year', 'power', 'kilometer', 'number_of_pictures', 'postal_code']
numeric_cols_index = [0,2,4,6,7,11]
categorical_cols_index = [1,3,5,7,9,10,11]
# Muestra gráficos para visualizar la distribución de los datos
# --------------------Descomenta el ciclo for para mostrar los gráficos.------------------------
# for i in numeric_cols_index[:-2]:
#     traza_plots_condicionados(df.iloc[:,i], nombres_columnas[i])

# Porcentaje de anomalos en "price"

# Calcula los niveles de los bigotes de boxplot de la columna "price"
up_lvl_price, low_lvl_price = iqr_calc(df,'price')
# Muestra los valores de los bigotes
print(up_lvl_price, low_lvl_price)
# Muestra un boxplot del primer cuartil de "price"
# df[df['price'] < df['price'].quantile(0.25)].boxplot('price') #--> Descomenta para mostrar el gráfico
# -------------- Quita el "#" de "plt.show()" para mostrar el gráfico
# plt.show()                                                    #--> Descomenta para mostrar el gráfico
# Muestra un boxplot de datos menores a 200 en "price".
# df[df['price'] < 200].boxplot('price')                        #--> Descomenta para mostrar el gráfico
# plt.show()
# Muestra un describe de "kilometer" para la columna de "price" menor al primer cuartil
df[df['price'] < df['price'].quantile(0.25)]['kilometer'].describe()
# Muestra el porcentaje de datos por debajo del primer cuartil
df[(df['price'] < df['price'].quantile(0.25))]['price'].count() / df.shape[0] * 100
# Muestra el porcentaje de datos por encima del bigote superior
df[df['price'] > up_lvl_price]['price'].count() / df.shape[0]

# Porcentaje de anomalos en "registration_year".

# Calcula los bigotes de la columna 'registration_year'
up_reg_year, low_reg_year = iqr_calc(df, 'registration_year')
# Muestra el porcentaje de vehiculos con registration_year anomalo
print(
    "Porcentaje de vehículos con 'registration_year' anomalo: ", 
    (np.round(df.query("registration_year < @low_reg_year or registration_year > @up_reg_year")['registration_year'].count() / df.shape[0] * 100,3)),"%"
    )

# Porcentaje de datos anomalos en "power".

# Calcula los niveles de los bigotes de la columna "power"
up_lvl_power, low_lvl_power = iqr_calc(df,'power')
#Muestra los niveles de los bigotes.
print("Nivel superior:", up_lvl_power, "Nivel inferior:",low_lvl_power)
# Porcentaje de vehiculos por encima del bigote superior.
print("Porcentaje de vehiculos por encima del bigote superior:",np.round(df[df['power'] > up_lvl_power]['power'].count() / df.shape[0] * 100,3),"%")
# Porcentaje de vehiculos por debajo del primer cuartil.
print("Porcentaje de vehiculos por debajo del primer valor del primer cuartil:",np.round(df[df['power'] < df['power'].quantile(0.25)]['power'].count() / df.shape[0] * 100,3),"%")
# Muestra el valor del primer cuartil
print("Valor de primer cuartil:", df['power'].quantile(0.25))

# Porcentaje de datos anomalos en "kilometer"

# Calcula el valor de los bigotes del boxplot de "kilometer"
up_lvl_kilometer, low_lvl_kilometer = iqr_calc(df,'kilometer')
# Muestra el valor del bigote inferior.
print(low_lvl_kilometer)
# Muestra el porcentaje de valores por debajo del bigote inferior.
print("Porcentaje de valores por debajo del bigote inferior:", np.round(df[df['kilometer']<low_lvl_kilometer]['kilometer'].count()/df.shape[0]*100,3))
