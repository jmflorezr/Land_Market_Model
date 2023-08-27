# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 12:25:59 2023

@author: Gustavo Bobadilla
"""

import pandas as pd

# Importar listado de municipios con precios y su respectivo año

ruta_actPx= "C:\\Users\\User\\OneDrive - JEP Colombia\\UPRA\\1 Agosto\\Tierras\\Actualizacion Precios.xlsx"
hoja_munPx = 'Municipios DIVIPOLA'
dfmunPx = pd.read_excel(ruta_actPx, sheet_name=hoja_munPx)
print(dfmunPx.head())



# Importar factores de actualización de precios

ruta_actPx= "C:\\Users\\User\\OneDrive - JEP Colombia\\UPRA\\1 Agosto\\Tierras\\Actualizacion Precios.xlsx"
hoja_IPC = 'IPC'
dfIPC = pd.read_excel(ruta_actPx, sheet_name=hoja_IPC)
print(dfIPC.head())


# Leer archivo de precios

ruta_Px = "C:\\Users\\User\\OneDrive - JEP Colombia\\UPRA\\1 Agosto\\Tierras\\Precio_Merc_Final_V2.xlsx"
dfPx = pd.read_excel(ruta_Px, engine='openpyxl', header=0)

# Identificar rangos de precios

rangos_px = dfPx.groupby(['cod_precios', 'rango_precios']).size().reset_index(name='count')
print(rangos_px)

# Eliminar códigos de rangos no válidos

cod_eliminar = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
rangos_px_dep = rangos_px.drop(cod_eliminar)
print(rangos_px_dep)

#Homogenizar los valores de los rangos para su procesamiento posterior

nuevo_valor = 'Mayor que 0 - hasta 1'
fila_indice = 0
columna_nombre = 'rango_precios'
rangos_px_dep.at[fila_indice, columna_nombre] = nuevo_valor
print(rangos_px_dep)

nuevo_valor = 'Mayor que 1000 - hasta 1000'
fila_indice = 25
columna_nombre = 'rango_precios'
rangos_px_dep.at[fila_indice, columna_nombre] = nuevo_valor
print(rangos_px_dep)


# Dividir la columna rango_precios en su valor mínimo y valor máximo

rangos_px_dep[['rango_min', 'rango_max']] = rangos_px_dep['rango_precios'].str.split('-', expand=True)

rangos_px_dep['rango_min'] = rangos_px_dep['rango_min'].str.replace(r'[^\d]', '', regex=True)
rangos_px_dep['rango_max'] = rangos_px_dep['rango_max'].str.replace(r'[^\d]', '', regex=True)

rangos_px_dep['rango_min'] = pd.to_numeric(rangos_px_dep['rango_min'])
rangos_px_dep['rango_max'] = pd.to_numeric(rangos_px_dep['rango_max'])

px_exp = 1000000

rangos_px_dep['rango_min'] = rangos_px_dep['rango_min'] * px_exp
rangos_px_dep['rango_max'] = rangos_px_dep['rango_max'] * px_exp

print(rangos_px_dep)

# definir clase clase de cada rango

rangos_px_dep['clase'] = (rangos_px_dep['rango_min'] + rangos_px_dep['rango_max']) / 2
rangos_px_dep['clase'] = pd.to_numeric(rangos_px_dep['clase'])
print(rangos_px_dep)




# Definir las categorías de rango por cada año hasta 2023 en un solo df

repeticiones_rangos = []

for año in dfIPC['Año']:
    repeticiones_rangos.append(rangos_px_dep.assign(Año=año))

px_actualizados = pd.concat(repeticiones_rangos, ignore_index=True)


columnas_seleccionadas = ['Año', 'Factor2023']
px_actualizados = pd.merge(px_actualizados, dfIPC[columnas_seleccionadas], left_on='Año', right_on='Año', how='left')

print(px_actualizados)




# Crear las nuevas columnas multiplicando por factor2023

px_actualizados['clase2023'] = px_actualizados['clase'] * px_actualizados['Factor2023']

print(px_actualizados)




# Importar tabla de rangos

ruta_actPx= "C:\\Users\\User\\OneDrive - JEP Colombia\\UPRA\\1 Agosto\\Tierras\\Actualizacion Precios.xlsx"
hoja_RangosPx = 'RangosPx'
dfRangosPx = pd.read_excel(ruta_actPx, sheet_name=hoja_RangosPx)
print(dfRangosPx.head())



# Encontrar el cod_precios correspondiente
def encontrar_cod_precio(row):
    filtro = (row['clase2023'] >= dfRangosPx['ValMin']) & (row['clase2023'] <= dfRangosPx['ValMax'])
    cod_precio = dfRangosPx.loc[filtro, 'cod_precios'].values
    return cod_precio[0] if len(cod_precio) > 0 else None

px_actualizados['cod_precio_v'] = px_actualizados.apply(encontrar_cod_precio, axis=1)
print(px_actualizados)



# Actualizar tabla de px de 2023 con los valores de los rangos originales

columnas_seleccionadas = ['cod_precios', 'rango_precios_v', 'clase_precios_v', 'ValMin', 'ValMax']
px_actualizados = pd.merge(px_actualizados, dfRangosPx[columnas_seleccionadas], left_on='cod_precio_v', right_on='cod_precios', how='left')
px_actualizados = px_actualizados.drop('cod_precios_y', axis=1)
print(px_actualizados)


path_to_save = "C:\\Users\\User\\OneDrive - JEP Colombia\\UPRA\\1 Agosto\\Tierras\\Resultados_Px_Actualizados.xlsx"
px_actualizados.to_excel(path_to_save, index=False)



# Eliminar filas de la tabla de px on códigos de px no procedentes


indices_deseados  = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
valores_identificados = rangos_px.loc[indices_deseados, 'rango_precios']

dfPx_2023 = dfPx[~dfPx['rango_precios'].isin(valores_identificados)]
print(dfPx_2023)




# Complementar tabla de precios con año de precio y factores de actualización

columnas_seleccionadas = ['CodMun', 'AñoPrecio']
dfPx_2023 = pd.merge(dfPx_2023, dfmunPx[columnas_seleccionadas], left_on='cod_dane_mpio', right_on='CodMun', how='left')
dfPx_2023.drop(columns=['CodMun'], inplace=True)
print(dfPx_2023)


# Actualizar tabla de px con los rangoas actualizados a 2023

dfPx_2023['llave'] = dfPx_2023['AñoPrecio'].astype(str) + dfPx_2023['cod_precios']
px_actualizados['llave'] = px_actualizados['Año'].astype(str) + px_actualizados['cod_precios_x']

dfPx_2023 = dfPx_2023.merge(px_actualizados[['llave', 'cod_precio_v', 'rango_precios_v', 'clase_precios_v', 'ValMin', 'ValMax']], on='llave', how='left')

dfPx_2023 = dfPx_2023.drop('llave', axis=1)

print(dfPx_2023)


# Graficas 


rangos_px_corr = dfPx_2023.groupby(['cod_precios', 'rango_precios']).size().reset_index(name='count1')
print(rangos_px)

rangos_px_corr_año = dfPx_2023.groupby(['cod_precios', 'rango_precios', 'AñoPrecio']).size().reset_index(name='count2')
print(rangos_px)

rangos_px_2023 = dfPx_2023.groupby(['cod_precio_v', 'rango_precios_v']).size().reset_index(name='count3')
print(rangos_px)



dfRangosPx  = dfRangosPx.merge(rangos_px_corr, on='cod_precios', how='left')
dfRangosPx = dfRangosPx.rename(columns={'count1': 'cant_PxCorr'})
dfRangosPx = dfRangosPx.drop('rango_precios', axis=1)
print(dfRangosPx)


dfRangosPx = dfRangosPx.merge(rangos_px_2023, left_on='cod_precios', right_on='cod_precio_v', how='left')
dfRangosPx = dfRangosPx.rename(columns={'count3': 'cant_Px2023'})
dfRangosPx = dfRangosPx.drop('cod_precio_v', axis=1)
dfRangosPx = dfRangosPx.drop('rango_precios_v_y', axis=1)
print(dfRangosPx)

dfRangosPx['cant_PxCorr'].fillna(0, inplace=True)
dfRangosPx['cant_Px2023'].fillna(0, inplace=True)
print(dfRangosPx)



import matplotlib.pyplot as plt



clase_precios_v = dfRangosPx['clase_precios_v']
cant_PxCorr = dfRangosPx['cant_PxCorr']
cant_Px2023 = dfRangosPx['cant_Px2023']

clase_precios_v_millions = clase_precios_v / 1000000

plt.figure(figsize=(9, 4))

plt.scatter(cant_PxCorr, clase_precios_v_millions, label='Predios a Px Corrientes', color='blue', marker='o')
plt.scatter(cant_Px2023, clase_precios_v_millions, label='Predios a Px 2023', color='red', marker='x')

plt.xlabel('Cantidad de Predios')
plt.ylabel('Valor Promedio de Predios (millones)')
plt.title('Relación entre Cantidad de Predios y Valor Promedio')

plt.legend()

plt.grid(True)
plt.show()


import numpy as np


clase_precios_v = dfRangosPx['clase_precios_v']
cant_PxCorr = dfRangosPx['cant_PxCorr']
cant_Px2023 = dfRangosPx['cant_Px2023']

clase_precios_v_millions = clase_precios_v / 1000000

ind = np.arange(len(clase_precios_v))

width = 0.35

fig, ax = plt.subplots(figsize=(9, 4))

rects1 = ax.bar(ind - width/2, cant_PxCorr, width, label='Predios a Px Corrientes')
rects2 = ax.bar(ind + width/2, cant_Px2023, width, label='Predios a Px de 2023')

ax.set_xlabel('Valor Promedio de Predios (millones)')
ax.set_ylabel('Cantidad de Predios')
ax.set_title('Cantidad de Predios a Precios Corrientes y Precios de 2023 por Valor Promedio')
ax.set_xticks(ind)
ax.set_xticklabels(clase_precios_v_millions, rotation=45, ha="right")

ax.legend()

plt.show()
