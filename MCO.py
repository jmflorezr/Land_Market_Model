# -*- coding: utf-8 -*-
"""
Última actualización: 18 Sep 2023 08:59 PM
@autor: JULIAN FLOREZ
@colaborador: GUSTAVO BOBADILLA
"""

import pandas as pd

##################################################################################################
#Modelo para el año 2016 sin serie de tiempo
# Definimos las columnas que queremos seleccionar
# Obtener el índice

# 1. Leer el archivo de Excel tabla Maestra
# Ruta del archivo
ruta = "C:\\Users\\User\\OneDrive - Unidad de Planificación Rural Agropecuaria - UPRA\\1 Agosto\\Modelos\\Tabla_Maestra.xlsx"
nombre_de_hoja = 'MCO'
df1 = pd.read_excel(ruta, sheet_name=nombre_de_hoja, engine='openpyxl', header=0)

columnas_df1 = df1.columns
df1.info()

#ruta = "C:\\Users\\User\\OneDrive - Unidad de Planificación Rural Agropecuaria - UPRA\\1 Agosto\\Modelos\\MunicipiosExcluidos.xlsx"
#nombre_de_hoja = 'MunicipiosExcluidos'
#MunicipiosExcluidos = pd.read_excel(ruta, sheet_name=nombre_de_hoja, engine='openpyxl', header=0)

#codigos_excluidos = MunicipiosExcluidos['COD_MUN_EXCL'].tolist()
#mascara = ~df1['COD_MPIO'].isin(codigos_excluidos)
#df1 = df1[mascara]



columnas_a_eliminar = ['COD_MPIO', 'DEPARTAMENTO', 'MUNICIPIO', 'CategorIa_de_ruralidad',
                       'IRV_2016', 
                      'TASA_CREC_AR_CULT_PERM_2016',
                       'TASA_CREC_AR_CULT_TRANSIT_2016'
                       ]

# Tomamos estas columnas del dataframe df1
df100 = df1.drop(columns=columnas_a_eliminar)

# # Eliminar las columnas especificadas del dataframe df100
# df101 = df100.drop(columns=["CategorIa_de_ruralidad"])

df101 = df100
df101.info()

# Eliminar las filas con valores nan en la columna especificada

df101 = df101.dropna(subset=["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016"])
df101['Area_Permanentes_2016'] = df101['Area_Permanentes_2016'].fillna(0)
df101['Area_Transitorios_2016'] = df101['Area_Transitorios_2016'].fillna(0)
df101["CPerm%2016"] = df101["CPerm%2016"].fillna(0)
df101["CTransi2016"] = df101["CTransi2016"].fillna(0)
df101['Pot2016'] = df101['Pot2016'].fillna(0)
df101 = df101.dropna(subset=["Indice_de_rendimiento_PromNal_MáxNal"])
#Idf101["TASA_CREC_AR_CULT_PERM_2016"] = df101["TASA_CREC_AR_CULT_PERM_2016"].fillna(0)
#df101["TASA_CREC_AR_CULT_TRANSIT_2016"] = df101["TASA_CREC_AR_CULT_TRANSIT_2016"].fillna(0)

df101.info()

#Matrix de correlacion 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df101.dropna(inplace=True)

# Descripción básica de los datos
print(df101.describe())

# Matriz de correlación
corr_matrix = df101.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Matriz de Correlación")
plt.show()

#Modelo econométrico por Mínimos Cuadrados Ordinarios (MCO)
import statsmodels.api as sm

# Definir la variable dependiente y las variables independientes
Y = df101["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016"]
X = df101.drop(columns=["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016"])

# # Si la columna cero es el número del municipio y no es una variable independiente, la eliminamos
# X = X.drop(X.columns[0], axis=1)

# Añadir una constante (intercepto) al modelo
X = sm.add_constant(X)

# Ajustar el modelo
model = sm.OLS(Y, X).fit()

# Mostrar los resultados==Premodelo
print(model.summary())

#Ajustes modelo en variables

columnas_a_seleccionar = [0, 2, 8, 
                           15, 17,
                          20, 29,
                          34, 35]

# Creamos el nuevo DataFrame df200
df200 = df101.iloc[:, columnas_a_seleccionar]
# Definir la variable dependiente y las variables independientes
Y = df200["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016"]
X = df200.drop(columns=["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016"])

# # Si la columna cero es el número del municipio y no es una variable independiente, la eliminamos
# X = X.drop(X.columns[0], axis=1)

# Añadir una constante (intercepto) al modelo
X = sm.add_constant(X)

# Ajustar el modelo
model = sm.OLS(Y, X).fit()

# Mostrar los resultados
print(model.summary())



# Definir la variable dependiente y las variables independientes
Y = df200["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016"]
X = df200.drop(columns=["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016"])

# Si la columna cero es el número del municipio y no es una variable independiente, elimínala
# Descomenta la siguiente línea si quieres eliminar la columna cero
# X = X.drop(X.columns[0], axis=1)

# Nota: NO estamos añadiendo una constante (intercepto) al modelo esta vez

# Ajustar el modelo
model_without_intercept = sm.OLS(Y, X).fit()

# Mostrar los resultados
print(model_without_intercept.summary())


#Pruebas
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# 1. Normalidad de los errores
residuals = model.resid
sm.qqplot(residuals, line='45')
plt.title('Q-Q plot para los residuos del modelo')
plt.show()

# 2. Multicolinealidad (VIF)
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# Un valor de VIF por encima de 5-10 indica problemas de multicolinealidad.

# 3. Homocedasticidad (Breusch-Pagan test)
bp_test = het_breuschpagan(residuals, model.model.exog)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print(dict(zip(labels, bp_test)))

# Un valor p bajo en la prueba de Breusch-Pagan indica heterocedasticidad.