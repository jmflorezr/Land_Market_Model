# -*- coding: utf-8 -*-

"""

Creado el miércoles 23 de agosto de 2023 a las 18:48
@author: GUSTAVO BOBADILLA

"""


import pandas as pd

# Crear el DataFrame con los datos panel de los precios de la tierra.
data_precios = {'año_precio': [2014, 2014, 2015, 2015, 2016, 2016, 2017, 2017, 2018, 2018, 2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023],
                'municipio': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
                'precio_original': [100, 120, 110, 130, 120, 140, 130, 150, 140, 160, 150, 170, 160, 180, 170, 190, 180, 200, 190, 210]}
df_precios_tierra = pd.DataFrame(data_precios)

# Tasas para la actualización del precio. Inicialmente se usa el IPC para cada año t-1.
tasa_actualizacion_precio = {
    2014: 0.0677,
    2015: 0.0575,
    2016: 0.0409,
    2017: 0.0318,
    2018: 0.0380,
    2019: 0.0161,
    2020: 0.0562,
    2021: 0.1312,
    2022: 0.0668,
    2023: 0.0  # Tasa de 0 para 2023 (sin actualización)
}

# Función para calcular el precio actualizado año tras año
def calcular_precio_actualizado(row):
    precio_actualizado = row['precio_original']
    for año in range(row['año_precio'], 2023):
        tasa = tasa_actualizacion_precio[año]
        precio_actualizado *= (1 + tasa)
    return precio_actualizado

# Aplicar la función para calcular el precio actualizado y crear una nueva columna
df_precios_tierra['precio_actualizado'] = df_precios_tierra.apply(calcular_precio_actualizado, axis=1)

# Mostrar el DataFrame actualizado
print(df_precios_tierra)