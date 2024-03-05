# -*- coding: utf-8 -*-
"""
Última actualización: 11 Oct 2023 08:59 PM
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
ruta = "C:\\Users\\JULIAN FLOREZ\\Downloads\\Variables\\Copia de 20231116_DTR_TABLA_MAESTRA_INF_REGISTRAL.xlsx"
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



columnas_a_eliminar = ['DINAMICA_SNR', 'COD_MPIO', 'DEPARTAMENTO', 'MUNICIPIO', 'CategorIa_de_ruralidad',
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
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

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
#X = X.drop(X.columns[0], axis=1)

# Añadir una constante (intercepto) al modelo
#X = sm.add_constant(X)

# Ajustar el modelo
model = sm.OLS(Y, X).fit()

# Mostrar los resultados==Premodelo
print(model.summary())

#Ajustes modelo en variables

columnas_a_seleccionar = [0, 2, 3,
                          8, 10, 15, 16, 17,
                          29]

# Creamos el nuevo DataFrame df200
df101_filtrado = df101[df101['PREDIOS_RURALES_2016'] >= 0]

df200 = df101_filtrado.iloc[:, columnas_a_seleccionar]

# Definir la variable dependiente y las variables independientes
Y = df200["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016"]
X = df200.drop(columns=["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016"])

# # Si la columna cero es el número del municipio y no es una variable independiente, la eliminamos
#X = X.drop(X.columns[0], axis=1)

# Añadir una constante (intercepto) al modelo
#X = sm.add_constant(X)

# Ajustar el modelo
model = sm.OLS(Y, X).fit()

# Mostrar los resultados==Premodelo
print(model.summary())

#Ajustes modelo en variables

columnas_a_seleccionar = [0, 2, 3,
                          8, 10, 12, 14, 15, 16, 17,
                          21, 29, 32, 35]

# Creamos el nuevo DataFrame df200
df101_filtrado = df101[df101['PREDIOS_RURALES_2016'] >= 0]

df200 = df101_filtrado.iloc[:, columnas_a_seleccionar]

# Lista de columnas de interés
columnas = [
    'PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016',
    'PREDIOS_RURALES_2016',
    'Has._Coca_2016',
    'Victimas_2016',
    'POBLACION_CP_Y_RURAL_DISP_2016',
    'Actividades_Secundarias_2016',
    'Valor_Agregado_2016',
    'Participacion_Agregado_2016',
    'Pot2016',
    'informalidad_2014',
    'Indice_de_rendimiento_PromNal_MáxNal',
    'Petracion_de_la_banda_ancha',
    'Pobreza_IPM',
    'INDICE ENVEJECIMIENTO RURAL_2016'
]

# Obtener estadísticas descriptivas para cada columna
estadisticas_descriptivas = df200[columnas].describe()


# Generar gráficos para cada columna
for columna in columnas:
    plt.figure(figsize=(10, 6))
    plt.title(f"Histograma de {columna}")
    df200[columna].hist(bins=20)
    plt.xlabel(columna)
    plt.ylabel('Frecuencia')
    plt.grid(False)
    plt.show()


# Creamos el nuevo DataFrame df200
df200 = df200[df200['informalidad_2014'] >= 0]

# Obtener estadísticas descriptivas para cada columna
estadisticas_descriptivas = df200[columnas].describe()

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import statsmodels.formula.api as smf


# Selección de variables
independent_vars = df200.columns.drop('PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016')
X = df200[independent_vars]
y = df200['PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016']

# Transformación logarítmica de los datos
high_vif_columns = ['PREDIOS_RURALES_2016', 'POBLACION_CP_Y_RURAL_DISP_2016', 
                    'informalidad_2014', 'Indice_de_rendimiento_PromNal_MáxNal', 'Pobreza_IPM', 
                    'INDICE ENVEJECIMIENTO RURAL_2016']
y_transformed = np.log1p(y)
X_transformed = X.copy()
X_transformed[high_vif_columns] = np.log1p(X_transformed[high_vif_columns])

# Eliminación de variables con alto VIF y no significativas
columns_to_remove = ['PREDIOS_RURALES_2016', 'Has._Coca_2016', 'POBLACION_CP_Y_RURAL_DISP_2016', 
                     'informalidad_2014', 'Indice_de_rendimiento_PromNal_MáxNal', 
                     'Pobreza_IPM', 'INDICE ENVEJECIMIENTO RURAL_2016', 'Valor_Agregado_2016', 'Actividades_Secundarias_2016']
X_transformed_reduced = X_transformed.drop(columns=columns_to_remove)

# División de los datos en conjuntos de entrenamiento y prueba
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_transformed_reduced, y_transformed, test_size=0.2, random_state=42)

# Modelo de mínimos cuadrados ponderados (WLS)
ols_model = sm.OLS(y_train_reduced, sm.add_constant(X_train_reduced)).fit()
ols_residuals = ols_model.resid
weights = 1 / (ols_residuals**2)
model = sm.WLS(y_train_reduced, sm.add_constant(X_train_reduced), weights=weights).fit()

# Resultados
print(model.summary())

#Pruebas
import matplotlib.pyplot as plt
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


# Prueba de Multicolinealidad (VIF)
vif_data = pd.DataFrame()
vif_data['feature'] = X_transformed_reduced.columns
vif_data['VIF'] = [variance_inflation_factor(X_transformed_reduced.values, i) for i in range(len(X_transformed_reduced.columns))]
print('VIF data:', vif_data)



# Un valor de VIF por encima de 5-10 indica problemas de multicolinealidad.

# 3. Homocedasticidad (Breusch-Pagan test)
bp_test = het_breuschpagan(residuals, model.model.exog)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print(dict(zip(labels, bp_test)))

##################################################



# import numpy as np
# # Lista de columnas a las que se les aplicará el logaritmo natural
# columnas_log = [
#     'PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016',
#     'PREDIOS_RURALES_2016',
#     'POBLACION_CP_Y_RURAL_DISP_2016'
            
# ]

# # Aplicar el logaritmo natural y actualizar el dataframe
# for columna in columnas_log:
#     # Asegurarse de que no hay valores cero o negativos antes de aplicar el logaritmo
#     if (df200[columna] <= 0).any():
#         print(f"Advertencia: La columna '{columna}' contiene valores cero o negativos que no son válidos para el logaritmo.")
#     else:
#         df200[columna] = np.log(df200[columna])

# # Obtener estadísticas descriptivas para cada columna
# estadisticas_descriptivas = df200[columnas].describe()

# #PCA
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# # Seleccionar las variables independientes
# variables_independientes = [
#     'PREDIOS_RURALES_2016',
#     'Has._Coca_2016',
#     'Victimas_2016',
#     'POBLACION_CP_Y_RURAL_DISP_2016',
#     'Actividades_Secundarias_2016',
#     'Valor_Agregado_2016',
#     'Participacion_Agregado_2016',
#     'Pot2016',
#     'informalidad_2014',
#     'Indice_de_rendimiento_PromNal_MáxNal',
#     'Petracion_de_la_banda_ancha',
#     'Pobreza_IPM',
#     'INDICE ENVEJECIMIENTO RURAL_2016'
# ]

# # Separar la variable dependiente
# X = df200[variables_independientes]

# # Normalizar las variables independientes
# scaler = StandardScaler()
# X_normalizado = scaler.fit_transform(X)

# # Aplicar PCA
# pca = PCA(n_components=5)
# componentes_principales = pca.fit_transform(X_normalizado)

# # Crear un DataFrame con los componentes principales
# df_pca = pd.DataFrame(data=componentes_principales, 
#                       columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

# # Varianza explicada por cada componente
# varianza_explicada = pca.explained_variance_ratio_
# print("Varianza explicada por cada componente:", varianza_explicada)

# from mpl_toolkits.mplot3d import Axes3D

# # Graficar los tres primeros componentes en un gráfico 3D
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df_pca['PC1'], df_pca['PC2'], df_pca['PC3'])
# ax.set_xlabel('Componente Principal 1 (PC1)')
# ax.set_ylabel('Componente Principal 2 (PC2)')
# ax.set_zlabel('Componente Principal 3 (PC3)')
# ax.set_title('PCA - Componentes 1, 2 y 3')
# plt.show()


# # Repetir para otros pares de componentes según sea necesario

# # Varianza explicada por cada componente
# varianza_explicada = pca.explained_variance_ratio_

# # Imprimir la varianza explicada
# print("Varianza explicada por cada componente:", varianza_explicada)

# # Sumar acumulativamente la varianza explicada
# varianza_acumulada = np.cumsum(varianza_explicada)

# # Graficar la varianza explicada y la varianza acumulada
# plt.figure(figsize=(8, 6))
# plt.bar(range(1, len(varianza_explicada) + 1), varianza_explicada, alpha=0.5, align='center', label='Varianza individual explicada')
# plt.step(range(1, len(varianza_acumulada) + 1), varianza_acumulada, where='mid', label='Varianza acumulada explicada')
# plt.ylabel('Proporción de Varianza Explicada')
# plt.xlabel('Componentes Principales')
# plt.legend(loc='best')
# plt.title('Varianza Explicada por Diferentes Componentes Principales')
# plt.show()


# # Obtener las cargas (loadings) de los componentes principales
# loadings = pca.components_

# # Convertir las cargas a un DataFrame para una mejor visualización
# df_loadings = pd.DataFrame(loadings.T, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=variables_independientes)

# # Mostrar las cargas de los componentes
# print(df_loadings)

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Visualizar las contribuciones de las variables a los componentes principales
# plt.figure(figsize=(12, 8))
# sns.heatmap(df_loadings, annot=True, cmap='coolwarm')
# plt.title('Cargas de las Variables en los Componentes Principales')
# plt.show()

# # Eliminar la columna 'Pot2016' directamente en df200
# #df200.drop(columns=['Pot2016'], inplace=True)

# # Lista de columnas de interés
# columnas = [
#     'PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016',
#     'PREDIOS_RURALES_2016',
#     'Has._Coca_2016',
#     'Victimas_2016',
#     'POBLACION_CP_Y_RURAL_DISP_2016',
#     'Actividades_Secundarias_2016',
#     'Valor_Agregado_2016',
#     'Participacion_Agregado_2016',
#     'Pot2016',
#     'informalidad_2014',
#     'Indice_de_rendimiento_PromNal_MáxNal',
#     'Petracion_de_la_banda_ancha',
#     'Pobreza_IPM',
#     'INDICE ENVEJECIMIENTO RURAL_2016'
# ]
# # Obtener estadísticas descriptivas para cada columna
# estadisticas_descriptivas = df200[columnas].describe()


# # Generar gráficos para cada columna
# for columna in columnas:
#     plt.figure(figsize=(10, 6))
#     plt.title(f"Histograma de {columna}")
#     df200[columna].hist(bins=20)
#     plt.xlabel(columna)
#     plt.ylabel('Frecuencia')
#     plt.grid(False)
#     plt.show() 

   
# # Definir la variable dependiente y las variables independientes
# Y = df200["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016"]
# X = df200.drop(columns=["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016"])

# # # Si la columna cero es el número del municipio y no es una variable independiente, la eliminamos
# # X = X.drop(X.columns[0], axis=1)

# # Añadir una constante (intercepto) al modelo
# #X = sm.add_constant(X)

# # # Ajustar el modelo
# model = sm.OLS(Y, X).fit()

# # # Mostrar los resultados
# print(model.summary())

# # Uso de errores estándar robustos para ajustar el modelo
# #model_robust = sm.OLS(Y, X).fit(cov_type='HC3')

# # Mostrar los resultados con errores estándar robustos
# #print(model_robust.summary())


# columnas_a_seleccionar = [6,
#                           10, 11, 12]

# # Creamos el nuevo DataFrame df200
# df101_filtrado = X[X['PREDIOS_RURALES_2016'] >= 0]

# X = df101_filtrado.iloc[:, columnas_a_seleccionar]

# # # Ajustar el modelo
# model = sm.OLS(Y, X).fit()

# # # Mostrar los resultados
# print(model.summary())


# # # Definir la variable dependiente y las variables independientes
# # Y = df200["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016"]
# # X = df200.drop(columns=["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016"])

# # Si la columna cero es el número del municipio y no es una variable independiente, elimínala
# # Descomenta la siguiente línea si quieres eliminar la columna cero
# # X = X.drop(X.columns[0], axis=1)

# # Nota: NO estamos añadiendo una constante (intercepto) al modelo esta vez

# # # Aquí usamos el inverso del cuadrado de la variable dependiente como pesos.
# # weights = 1 / X['Pobreza_IPM']**2

# # # Ajustar el modelo WLS
# # wls_model = sm.WLS(Y, X, weights=weights).fit()

# # # Mostrar los resultados del modelo WLS
# # print(wls_model.summary())

# # # Para usar errores estándar robustos con WLS
# # wls_model_robust = wls_model.get_robustcov_results(cov_type='HC3')

# # # Mostrar los resultados con errores estándar robustos
# # print(wls_model_robust.summary())

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import matplotlib.pyplot as plt



import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

# Carga de datos
file_path = 'C:\\Users\\JULIAN FLOREZ\\Downloads\\Prueba_IA_01.xlsx'  
excel_data = pd.read_excel(file_path)

# Selección de variables para el modelo, excluyendo la variable dependiente
independent_vars = excel_data.columns.drop('CAMBIO_DE_PROPIETARIO_2016')
X = excel_data[independent_vars]
y = excel_data['CAMBIO_DE_PROPIETARIO_2016']

# Transformación logarítmica de la variable dependiente y algunas variables independientes
high_vif_columns = ['PREDIOS_RURALES_2016', 'informalidad_2014', 
                    'Indice_de_rendimiento_PromNal_MáxNal', 'Pobreza_IPM', 
                    'INDICE ENVEJECIMIENTO RURAL_2016', 'POBLACION_CP_Y_RURAL_DISP_2016',
                    'Has._Coca_2016']
y_transformed = np.log1p(y)
X_transformed = X.copy()
X_transformed[high_vif_columns] = np.log1p(X_transformed[high_vif_columns])

#Eliminación de variables con alto VIF
columns_to_remove = ['Valor_Agregado_2016', 'Victimas_2016', 'PREDIOS_RURALES_2016',
                     'informalidad_2014', 'Indice_de_rendimiento_PromNal_MáxNal',
                     'Pobreza_IPM', 'INDICE ENVEJECIMIENTO RURAL_2016','Pot2016']
X_transformed_reduced = X_transformed.drop(columns=columns_to_remove)

# División de los datos en conjuntos de entrenamiento y prueba
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_transformed_reduced, y_transformed, test_size=0.2, random_state=42)

# Modelo de mínimos cuadrados ponderados (WLS)
ols_model = sm.OLS(y_train_reduced, sm.add_constant(X_train_reduced)).fit()
ols_residuals = ols_model.resid
weights = 1 / (ols_residuals**2)
model = sm.WLS(y_train_reduced, sm.add_constant(X_train_reduced), weights=weights).fit()

# Resultados del modelo
print(model.summary())

# Prueba de Normalidad de los errores (Shapiro-Wilk)
errors = y_test_reduced - model.predict(sm.add_constant(X_test_reduced))
shapiro_test = stats.shapiro(errors)
print('Shapiro-Wilk test:', shapiro_test)

# Prueba de Multicolinealidad (VIF)
vif_data = pd.DataFrame()
vif_data['feature'] = X_transformed_reduced.columns
vif_data['VIF'] = [variance_inflation_factor(X_transformed_reduced.values, i) for i in range(len(X_transformed_reduced.columns))]
print('VIF data:', vif_data)

# Prueba de Homocedasticidad (Breusch-Pagan)
_, bp_pvalue, _, _ = het_breuschpagan(errors, sm.add_constant(X_test_reduced))
print('Breusch-Pagan test p-value:', bp_pvalue)


###################################

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

#################################
import matplotlib.pyplot as plt

# Lista de columnas de interés
columnas = [
    'Has._Coca_2016',
    'POBLACION_CP_Y_RURAL_DISP_2016',
    'Actividades_Secundarias_2016',
    'Participacion_Agregado_2016',
    'Petracion_de_la_banda_ancha',
]

# Obtener estadísticas descriptivas para cada columna
estadisticas_descriptivas = X_train_reduced[columnas].describe()


# Generar gráficos para cada columna
for columna in columnas:
    plt.figure(figsize=(10, 6))
    plt.title(f"Histograma de {columna}")
    X_train_reduced[columna].hist(bins=20)
    plt.xlabel(columna)
    plt.ylabel('Frecuencia')
    plt.grid(False)
    plt.show()

