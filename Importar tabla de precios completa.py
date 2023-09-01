# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 21:18:09 2023

@author: Gustavo Bobadilla
"""

# Importar librerías

import pandas as pd

# Importar tabla de precios completa
file_path = r"C:\Users\User\OneDrive - Unidad de Planificación Rural Agropecuaria - UPRA\1 Agosto\Modelos\Precio_Merc_Final.txt"


# Cargar el archivo validando las diferentes codificaciones con el separador ;

encodings = ['utf-8', 'ISO-8859-1', 'cp1252']

for encoding in encodings:
    try:
        dfPx = pd.read_csv(file_path, sep=';', encoding=encoding)
        print("Importación exitosa con codificación:", encoding)
        break
    except UnicodeDecodeError:
        print("Error al intentar con codificación:", encoding)
    except pd.errors.EmptyDataError:
        print("No se encontraron datos en el archivo.")

# Identificar campos de la tabla de precios
info_dfPx = dfPx.info()
print(info_dfPx)

# Identificar rangos de precios
rangos_px = dfPx.groupby(['cod_precios', 'rango_precios']).size().reset_index(name='count')
print(rangos_px)