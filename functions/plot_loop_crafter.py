
# Libraries -----------------------------------------------------------------------------------
from matplotlib import pyplot as plt

# Functions -----------------------------------------------------------------------------------

def traza_plots_condicionados (columna, columna_nombre):
    """
    Función que recibe un series (columna) y traza un plot dependiendo de la cantidad de valores únicos en la columna.
    Si la columna tiene más de 10 valores únicos traza un boxplot, si no un histograma.
    """
    if columna.nunique() < 10:
        plt.bar(columna.unique(),columna.value_counts())
        plt.xticks(columna.unique())
        plt.title(f'Gráfico de barras de {columna_nombre}.')
        plt.xlabel(f'{columna_nombre}.')
        plt.ylabel('Cantidad.')
        plt.show()
        return
    else:
        plt.boxplot(columna, columna_nombre)
        plt.title(f'Boxplot de la distribución de {columna_nombre}.')
        plt.xlabel(f'{columna_nombre}.')
        plt.ylabel('Cantidad.')
        plt.show()
        return