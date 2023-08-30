# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:17:59 2023

@author: Gustavo Bobadilla
"""

import pandas as pd

# Importar listado de actualización del POT

# Ruta del archivo Excel y nombre de la hoja
ruta_POT = "C:\\Users\\JULIAN\\Downloads\\Variables\\Base maestra POT Colombia-Corte 31-07-2023_Para el DNP (1).xlsx"
hoja_POT = 'Hoja1'

# Leer el archivo Excel con la primera columna como texto
dfPOT = pd.read_excel(ruta_POT, sheet_name=hoja_POT, dtype={"COD MUNIC LLAVE": str})


# Convierte la columna "FECHA" al formato datetime y crea la variable Año

dfPOT['FECHA'] = pd.to_datetime(dfPOT['FECHA'], errors='coerce')

dfPOT['AÑO'] = dfPOT['FECHA'].dt.year

dfPOT['AÑO'].fillna('', inplace=True)

# Conteo

AÑOS_POT = dfPOT.groupby(['AÑO']).size().reset_index(name='count_anios_pot')

# Identificar si hay municipios repetidos

duplicated_counts = dfPOT['COD MUNIC LLAVE'].duplicated(keep=False)
duplicated_data = dfPOT[duplicated_counts]
value_counts = duplicated_data['COD MUNIC LLAVE'].value_counts()


# Obtener los codigos de municipio únicos
unique_cod_mun = dfPOT["COD MUNIC LLAVE"].unique()


# Crear nuevo df para las variables DUMMY que indican si se ha actualizado el POT

data = {
    "COD MUN": [],
    "2014": [],
    "2015": [],
    "2016": [],
    "2017": [],
    "2018": [],
    "2019": []
}

for cod_mun in unique_cod_mun:
    data["COD MUN"].append(cod_mun)
    data["2014"].append(0)
    data["2015"].append(0)
    data["2016"].append(0)
    data["2017"].append(0)
    data["2018"].append(0)
    data["2019"].append(0)

DUMMY_POT = pd.DataFrame(data)

print(DUMMY_POT)



# Actualizar el df DUMMY POT con las columnas AÑO y ESTADO

DUMMY_POT = DUMMY_POT.merge(dfPOT[['COD MUNIC LLAVE', 'AÑO', 'ESTADO']],
                             how='left', left_on='COD MUN', right_on='COD MUNIC LLAVE')

DUMMY_POT.drop(columns=['COD MUNIC LLAVE'], inplace=True)

# Convertir la columna "AÑO" a formato numérico
DUMMY_POT['AÑO'] = pd.to_numeric(DUMMY_POT['AÑO'], errors='coerce')

# Asignar el número cero a los valores NaN en la columna 'AÑO'
DUMMY_POT['AÑO'].fillna(0, inplace=True)

# Definir las condiciones y las columnas de los años
conditions = [
    DUMMY_POT['AÑO'] == 2013,
    DUMMY_POT['AÑO'] == 2014,
    DUMMY_POT['AÑO'] == 2015,
    DUMMY_POT['AÑO'] == 2016,
    DUMMY_POT['AÑO'] == 2017,
    DUMMY_POT['AÑO'] == 2018,
    DUMMY_POT['AÑO'] == 2019
]

year_columns = ["2014", "2014", "2015", "2016", "2017", "2018", "2019"]

# Iterar a través de las condiciones y columnas de los años
for i, condition in enumerate(conditions):
    # Asignar el valor 1 a la columna del año correspondiente y a las columnas de los años posteriores
    for column in year_columns[i:]:
        DUMMY_POT.loc[condition, column] = DUMMY_POT.loc[condition, column].fillna(0) + 1

# Definir la condición donde la columna 'AÑO' es igual a 2014
condition_2014 = DUMMY_POT['AÑO'] == 2014

# Lista de columnas de años donde queremos asignar el valor 2
year_columns = ["2014", "2015", "2016", "2017", "2018", "2019"]

# Utilizar loc para asignar el valor 2 a las columnas correspondientes donde la condición se cumpla
for column in year_columns:
    DUMMY_POT.loc[condition_2014, column] = 2

# Lista de columnas de años donde queremos aplicar las modificaciones
year_columns = ["2014", "2015", "2016", "2017", "2018", "2019"]

# Iterar sobre cada columna de año
for column in year_columns:
    # Cambiar los valores que son igual a 1 a 0
    condition_is_one = DUMMY_POT[column] == 1
    DUMMY_POT.loc[condition_is_one, column] = 0

    # Cambiar los valores que son mayores a 1 a 1
    condition_greater_one = DUMMY_POT[column] > 1
    DUMMY_POT.loc[condition_greater_one, column] = 1


