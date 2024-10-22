# Descripción del proyecto.

## Una empresa ficticia **"Rusty Bargain"** se dedica a la venta de autos y está desarrollando una aplicación, la cual tendrá el propósito de poder averiguar el valor en el mercado de tú vehículo, basándonos en algunas característcas del vehículo como: marca, modelo, año del modelo, tipo de combustible y si ha tenido reparaciones previas. Se evaluaron modelos de **Machine Learning** para elegir el mejor en términos de rápidez de entrenamiento y calidad de predicción.

# Motivación

## **Rusty Bargain** preocupado por sus clientes proporciona una inovadora aplicación para los mismos, pensando en lo complicado que puede llegar a ser determinar el valor justo de un vehículo en el mercado de automóviles usados, ya que depende de factores diversos. 

# Características del Dataset

## 
- DateCrawled: fecha en la que se descargó el perfil de la base de datos
- Price: precio (en euros)
- VehicleType: tipo de carrocería del vehículo
- RegistrationYear: año de matriculación del vehículo
- Gearbox: tipo de caja de cambios
- Power: potencia (CV)
- Model: modelo del vehículo
- Kilometer: kilometraje (medido en km de acuerdo con las especificidades regionales del conjunto de datos)
- RegistrationMonth: mes de matriculación del vehículo
- FuelType: tipo de combustible
- Brand: marca del vehículo
- Repaired: vehículo con o sin reparación
- DateCreated: fecha de creación del perfil
- NumberOfPictures: número de fotos del vehículo
- PostalCode: código postal del propietario del perfil (usuario)
- LastSeen: fecha de la última vez que el usuario estuvo activo

# Herramientas Utilizadas

## 
- Lenguage: Python 3.11.2
- Librerías:
	- Pandas: Manipulación de datos.
	- Numpy: Cálculos matemáticos.
	- re (Regular Expression operations): Preprocesar nombres de las columnas.
	- Category Encoders: Categorizar Características sin aumentar la dimensionalidad.
	- Matplotlib y Seaborn: Visualización de datos con gráficos.
	- Scikit-learn: Preparación de los datos, creación y evaluación de modelos.
	- Catboost, LightGBM, XGBoost: Modelos de Regresión con Potenciación de Gradiente.
	
# Proceso del proyecto

## 
	1. Carga y exploración de los datos.
		- Análisis inicial del conjunto de datos: revisión de la estructura, revisión de valores faltantes, tipos y distribución de los mismos, así como identificar qué datos no son necesarios para los modelos.
	2. Limpieza y preprocesamiento.
		- Eliminar columnas que no servirán para los modelos.
		- Ajustar los nombres de las columnas a un formato general (camel case)
		- Rellenar valores faltantes.
		- Eliminar valores anomalos.
		- Normalizar valores númericos.
	3. Búsqueda de modelos.
		- Creación de modelos y cálculo de su rendimiento (rmse)
			- Regresión Líneal.
			- XGBoost.
			- Bosque Aleatorio.
			- CatBoost.
			- LightGBM.
		- Elección del mejor modelo.
			- El modelo que tuviera el entrenamiento rápido y con los mejores resultados (rmse).
	4. Resultados.
		- El modelo final puede predecir el precio de un vehículo con un margen de error de 1800 euros.