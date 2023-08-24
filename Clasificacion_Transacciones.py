# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:53:08 2023
@author: JULIAN FLOREZ
"""

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 1. Leer el archivo de Excel y tomar las columnas "Predios" y "Transacciones"
df = pd.read_excel("C:\\Users\\JULIAN FLOREZ\\Downloads\\Variables\\Transacciones01.xls")[["Predios", "Transacciones", "Categoria"]]

# 2. Usar k-means para clasificar las Transacciones
kmeans = KMeans(n_clusters=6, random_state=0).fit(df[['Transacciones']])
df['cluster'] = kmeans.labels_

# 3. Asignar etiquetas a los clusters basados en los centroides
sorted_idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
label_map = {
    sorted_idx[0]: 'muy bajo',
    sorted_idx[1]: 'bajo',
    sorted_idx[2]: 'medio',
    sorted_idx[3]: 'alto'
}
df['clasificacion'] = df['cluster'].map(label_map)
df.drop('cluster', axis=1, inplace=True)

# 4. Visualizar la clasificación
colors = {'muy bajo':'blue', 'bajo':'green', 'medio':'red', 'alto':'orange'}
plt.figure(figsize=(10, 6))

# Dibuja cada punto de dato
for clasificacion, color in colors.items():
    subset = df[df['clasificacion'] == clasificacion]
    plt.scatter(subset['Predios'], subset['Transacciones'], s=50, c=color, label=str(clasificacion))

# Dibuja los centroides
plt.scatter(kmeans.cluster_centers_, kmeans.cluster_centers_, s=200, c='yellow', marker='X', label='Centroides')

plt.title('Clasificación de Transacciones usando K-means')
plt.xlabel('Predios')
plt.ylabel('Transacciones')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# 5. Exportar los resultados a un nuevo archivo de Excel
path_to_save = "C:\\Users\\JULIAN FLOREZ\\Downloads\\Variables\\Resultados_Transacciones.xlsx"
df.to_excel(path_to_save, index=False)
######################SOM
import numpy as np
import pandas as pd
from minisom import MiniSom
import matplotlib.pyplot as plt

# Suponiendo que df es tu DataFrame
df['Categoria'].replace({
    'Ciudades y aglomeraciones': 0,
    'Intermedio': 1,
    'Rural disperso': 2,
    'Rural': 3
}, inplace=True)

# Asegurándonos de que la columna "Categoria" es de tipo categórico
df['Categoria'] = df['Categoria'].astype('category')

data = df[['Predios', 'Transacciones', 'Categoria']].values

# Crear y entrenar el SOM
som = MiniSom(7, 7, 3, sigma=1.0, learning_rate=0.5)
som.train(data, 1000)

# Obtener las distancias para cada punto en el SOM
distances = np.array([som.distance_map()[som.winner(d)] for d in data])
labels = pd.cut(distances, 4, labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto'])
df['clasificacion'] = labels

# Visualizar el SOM
weights = som.get_weights()

# Normalizar los pesos a [0,1]
normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())

plt.figure(figsize=(10, 10))
for i in range(normalized_weights.shape[0]):
    for j in range(normalized_weights.shape[1]):
        plt.plot(i, j, 'o', markerfacecolor=normalized_weights[i, j, :], markersize=10)

plt.title('SOM después de 1000 iteraciones')
plt.show()

# Gráfica cartesiana de 'Predios' vs 'Transacciones' coloreada por 'clasificacion'
colors = {'Muy Bajo': 'red', 'Bajo': 'orange', 'Medio': 'green', 'Alto': 'magenta'}
plt.figure(figsize=(10, 10))
for label in df['clasificacion'].cat.categories:
    mask = df['clasificacion'] == label
    plt.scatter(df['Predios'][mask], df['Transacciones'][mask], c=colors[label], label=label)

plt.xlabel('Predios')
plt.ylabel('Transacciones')
plt.title('Gráfica de Predios vs Transacciones y segementada por clasificación ruralidad coloreada por clasificación')
plt.legend()
plt.show()

###################Precios
# 1. Leer el archivo de Excel y tomar las columnas "Predios" y "Transacciones"

ruta = "C:\\Users\\JULIAN FLOREZ\\Downloads\\Variables\\Precio_Merc_Final_V2.xlsx"
df1000 = pd.read_excel(ruta, engine='openpyxl', header=0)
###############Panel data

# 1. Leer el archivo de Excel tabla Maestra
# Ruta del archivo
ruta = "C:\\Users\\JULIAN FLOREZ\\Downloads\\Variables\\Tabla_Maestra.xlsx"

df1 = pd.read_excel(ruta, engine='openpyxl', header=0)
df2 = df1.iloc[:, :52]
df2 = df2.drop(columns=['DEPARTAMENTO', 'MUNICIPIO', 'Gini_2020'])
df3 = df2.dropna(subset=['PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2019'])


# Para la variable dependiente
df_predios_cambio = pd.melt(df3, id_vars=['COD_MPIO'], 
                            value_vars=['PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2014','PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2015', 'PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016',
                                        'PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2017', 'PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2018', 'PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2019'],
                            var_name='Year', value_name='Predios_Cambio')

df_predios_cambio['Year'] = df_predios_cambio['Year'].str.replace("PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_", "")
df_predios_cambio['Year'] = df_predios_cambio['Year'].astype(int)

# Para el índice GINI
df_gini = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Gini_2014','Gini_2015','Gini_2016','Gini_2017','Gini_2018','Gini_2019'], 
                  var_name='Year', value_name='Gini')
df_gini['Year'] = df_gini['Year'].str.replace("Gini_", "")
df_gini['Year'] = df_gini['Year'].astype(int)

df_PredRur = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['PREDIOS_RURALES_2014','PREDIOS_RURALES_2015','PREDIOS_RURALES_2016','PREDIOS_RURALES_2017','PREDIOS_RURALES_2018','PREDIOS_RURALES_2019'], 
                 var_name='Year', value_name='PRurales')
df_PredRur['Year'] =  df_PredRur['Year'].str.replace("PREDIOS_RURALES_", "")
df_PredRur['Year'] =  df_PredRur['Year'].astype(int)

df_Coca = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Has. Coca 2014','Has. Coca 2015','Has. Coca 2016','Has. Coca 2017','Has. Coca 2018','Has. Coca 2019'], 
                  var_name='Year', value_name='Coca')
df_Coca['Year'] = df_Coca['Year'].str.replace("Has. Coca ", "")
df_Coca['Year'] = df_Coca['Year'].astype(int)

df_Transit = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Area_Transitorios_2014','Area_Transitorios_2015','Area_Transitorios_2016','Area_Transitorios_2017','Area_Transitorios_2018','Area_Transitorios_2019'], 
                  var_name='Year', value_name='Transitorios')
df_Transit['Year'] = df_Transit['Year'].str.replace("Area_Transitorios_", "")
df_Transit['Year'] = df_Transit['Year'].astype(int)

df_Perm = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Area_Permanentes_2014','Area_Permanentes_2015','Area_Permanentes_2016','Area_Permanentes_2017','Area_Permanentes_2018','Area_Permanentes_2019'], 
                  var_name='Year', value_name='Permanentes')
df_Perm['Year'] = df_Perm['Year'].str.replace("Area_Permanentes_", "")
df_Perm['Year'] = df_Perm['Year'].astype(int)

df_vict = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Victimas_2014','Victimas_2015','Victimas_2016','Victimas_2017','Victimas_2018','Victimas_2019'], 
                  var_name='Year', value_name='Victimas')
df_vict['Year'] = df_vict['Year'].str.replace("Victimas_", "")
df_vict['Year'] = df_vict['Year'].astype(int)

df_poblrur = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['POBLACION_CP_Y_RURAL_DISP_2014','POBLACION_CP_Y_RURAL_DISP_2015','POBLACION_CP_Y_RURAL_DISP_2016','POBLACION_CP_Y_RURAL_DISP_2017','POBLACION_CP_Y_RURAL_DISP_2018','POBLACION_CP_Y_RURAL_DISP_2019'], 
                  var_name='Year', value_name='PoblRural')
df_poblrur['Year'] = df_poblrur['Year'].str.replace("POBLACION_CP_Y_RURAL_DISP_", "")
df_poblrur['Year'] = df_poblrur['Year'].astype(int)



df_panel = (df_gini.merge(df_Coca, on=['COD_MPIO', 'Year'])
                    .merge(df_PredRur, on=['COD_MPIO', 'Year'])
                    .merge(df_Transit, on=['COD_MPIO', 'Year'])
                    .merge(df_Perm, on=['COD_MPIO', 'Year'])
                    .merge(df_vict, on=['COD_MPIO', 'Year'])
                    .merge(df_poblrur, on=['COD_MPIO', 'Year'])
                    .merge(df_predios_cambio, on=['COD_MPIO', 'Year']))

df_panel = df_panel.sort_values(by=['COD_MPIO', 'Year'])
df_panel['Victimas'] = df_panel['Victimas'].fillna(0)

#Análisis descriptivo básico
print(df_panel[['Gini', 'Coca', 'PRurales', 'Transitorios', 'Permanentes', 'PoblRural', 'Predios_Cambio']].describe())


#Visualizar tendencias a lo largo del tiempo:
import matplotlib.pyplot as plt

# Predios_Cambio tendencias
df_panel.groupby('Year')['Predios_Cambio'].mean().plot(label='Predios_Cambio', linestyle='--')

# Variables independientes
df_panel.groupby('Year')['Gini'].mean().plot(label='Gini', linestyle='--')
df_panel.groupby('Year')['Coca'].mean().plot(label='Coca', linestyle='--')
df_panel.groupby('Year')['PRurales'].mean().plot(label='PRurales', linestyle='--')
df_panel.groupby('Year')['Transitorios'].mean().plot(label='Transitorios', linestyle='--')
df_panel.groupby('Year')['Permanentes'].mean().plot(label='Permanentes', linestyle='--')
df_panel.groupby('Year')['Victimas'].mean().plot(label='Victimas', linestyle='--')
df_panel.groupby('Year')['PoblRural'].mean().plot(label='PoblRural', linestyle='--')
plt.legend()
plt.title('Tendencia de variables a lo largo del tiempo')
plt.show()


#Visualizar distribuciones
# Predios_Cambio
df_panel['Predios_Cambio'].hist(bins=30, alpha=0.5, label='Predios_Cambio')

# Variables independientes
df_panel['Gini'].hist(bins=30, alpha=0.5, label='Gini')
df_panel['Coca'].hist(bins=30, alpha=0.5, label='Coca')
df_panel['PRurales'].hist(bins=30, alpha=0.5, label='PRurales')
df_panel['Transitorios'].hist(bins=30, alpha=0.5, label='Transitorios')
df_panel['Permanentes'].hist(bins=30, alpha=0.5, label='Permanentes')
df_panel['Victimas'].hist(bins=30, alpha=0.5, label='Victimas')
df_panel['PoblRural'].hist(bins=30, alpha=0.5, label='PoblRural')
plt.legend()
plt.title('Distribución de variables')
plt.show()

#Correlaciones:
correlations = df_panel[['Predios_Cambio', 'Gini', 'Coca', 'PRurales', 'Transitorios', 'Permanentes', 'Victimas', 'PoblRural']].corr()
print(correlations)

#Analizar valores perdidos
missing_data = df_panel[['Predios_Cambio', 'Gini', 'Coca', 'PRurales', 'Transitorios', 'Permanentes', 'Victimas', 'PoblRural']].isnull().sum()
print(missing_data)

#Modelo de datos de panel
from linearmodels.panel import PanelOLS

# Configura un multi-índice con COD_MPIO y Year
df_panel = df_panel.set_index(['COD_MPIO', 'Year'])

# Modelo de efectos fijos
formula = 'Predios_Cambio ~ 1 + Gini + Coca + PRurales + Transitorios + Permanentes + Victimas + PoblRural + EntityEffects'
model = PanelOLS.from_formula(formula, data=df_panel)
results = model.fit()
print(results)

formula = 'Predios_Cambio ~ 1 + Gini + PRurales + Permanentes + Victimas + PoblRural + EntityEffects'
model = PanelOLS.from_formula(formula, data=df_panel)
results = model.fit()
print(results)

formula = 'Predios_Cambio ~ 1 + Gini + PRurales + Victimas + PoblRural + EntityEffects'
model = PanelOLS.from_formula(formula, data=df_panel)
results = model.fit()
print(results)

#Modelo de efectos aleatorios
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# Preparación de los datos
df_panel = df_panel.reset_index()
dependent = df_panel['Predios_Cambio']
independent = df_panel[['Gini', 'Coca', 'PRurales', 'Transitorios', 'Permanentes', 'Victimas', 'PoblRural']]
independent = sm.add_constant(independent)  # Añadir constante

# Modelo
model_re = MixedLM(dependent, independent, groups=df_panel['COD_MPIO'])
result_re = model_re.fit()
print(result_re.summary())


##########################################################
#La prueba de Hausman
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats

# Configura un multi-índice con COD_MPIO y Year
df_panel = df_panel.set_index(['COD_MPIO', 'Year'])

# Efectos Fijos
formula_fe = 'Predios_Cambio ~ 1 + Gini + Coca + PRurales + Transitorios + Permanentes + Victimas + EntityEffects'
model_fe = PanelOLS.from_formula(formula_fe, data=df_panel)
results_fe = model_fe.fit()

# Efectos Aleatorios
formula_re = 'Predios_Cambio ~ 1 + Gini + Coca + PRurales + Transitorios + Permanentes + Victimas'
model_re = RandomEffects.from_formula(formula_re, data=df_panel)
results_re = model_re.fit()

# Prueba de Hausman
b = results_fe.params
B = results_re.params
v_b = results_fe.cov
v_B = results_re.cov

# fórmula para la estadística de prueba
chi2 = np.dot((b - B).T, np.linalg.inv(v_b - v_B).dot(b - B)) 

df = b.shape[0]  # grados de libertad
p_value = 1 - stats.chi2.cdf(chi2, df)

print('Chi-squared:', chi2)
print('p-value:', p_value)

if p_value < 0.05:
    print("La prueba de Hausman es significativa: prefiera el modelo de efectos fijos.")
else:
    print("La prueba de Hausman no es significativa: prefiera el modelo de efectos aleatorios.")

##################################################################################################
#Modelo para el año 2019 sin serie de tiempo

# Definimos las columnas que queremos seleccionar
column_indices = [0, 8, 15, 21, 27, 33, 39, 45, 47, 49, 50, 51, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 72, 73, 39]

# Tomamos estas columnas del dataframe df1
df100 = df1.iloc[:, column_indices]

# Eliminar las columnas especificadas del dataframe df100
df101 = df100.drop(columns=["Categoría_de_ruralidad", "Categoría_de_municipio_2022", "Sub_región", "COD_MPIO"])

# Eliminar las filas con valores nan en la columna especificada
df101 = df101.dropna(subset=["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2019"])

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
Y = df101["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2019"]
X = df101.drop(columns=["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2019"])

# # Si la columna cero es el número del municipio y no es una variable independiente, la eliminamos
# X = X.drop(X.columns[0], axis=1)

# Añadir una constante (intercepto) al modelo
X = sm.add_constant(X)

# Ajustar el modelo
model = sm.OLS(Y, X).fit()

# Mostrar los resultados
print(model.summary())

#Ajustes modelo en variables

columns_to_select = list(range(0, 7)) + [11, 16, 17, 28, 29]

# Creamos el nuevo DataFrame df200
df200 = df101.iloc[:, columns_to_select]
# Definir la variable dependiente y las variables independientes
Y = df200["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2019"]
X = df200.drop(columns=["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2019"])

# # Si la columna cero es el número del municipio y no es una variable independiente, la eliminamos
# X = X.drop(X.columns[0], axis=1)

# Añadir una constante (intercepto) al modelo
X = sm.add_constant(X)

# Ajustar el modelo
model = sm.OLS(Y, X).fit()

# Mostrar los resultados
print(model.summary())

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

#####################################################################################
########Analitica de datos

df100.info()

# Crear un diccionario con las categorías y sus valores correspondientes
categorias = {
    "#N/D": 0,
    "Rural disperso": 1,
    "Rural": 2,
    "Intermedio": 3,
    "Ciudades y aglomeraciones": 4
}

# Reemplazar los valores en la columna Categoría_de_ruralidad usando el diccionario
df100['Categoría_de_ruralidad'] = df100['Categoría_de_ruralidad'].replace(categorias)


# Crear un diccionario con las categorías y sus valores correspondientes
categorias = {
    "#N/D": 0,
    "Temprano": 1,
    "Robusto": 2,
    "Intermedio": 3
}

# Reemplazar los valores en la columna Entorno_del_desarrollo usando el diccionario
df100['Entorno_del_desarrollo'] = df100['Entorno_del_desarrollo'].replace(categorias)

#Datos Categoricos
df100['Categoría_de_ruralidad'] = df100['Categoría_de_ruralidad'].astype('category')
df100['Entorno_del_desarrollo'] = df100['Entorno_del_desarrollo'].astype('category')

df100.dropna(inplace=True)


# Realizamos la unión de los dataframes usando la columna indicada
merged_df = df100.merge(df, left_on='PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2019', right_on='Transacciones', how='left')

# Asignamos los valores de la columna 'clasificacion' del dataframe df a la nueva columna 'transac' en df100
df100['transac'] = merged_df['clasificacion']

# Crear un diccionario con las categorías y sus valores correspondientes
categorias = {
    "bajo": 0,
    "medio": 1,
    "alto": 2
}

# Reemplazar los valores en la columna Entorno_del_desarrollo usando el diccionario
df100['transac'] = df100['transac'].replace(categorias)

df100['transac'] = df100['transac'].astype('category')
df100.drop("PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2019", axis=1, inplace=True)

y = df100['transac']
X = df100.drop(columns = 'transac')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
sns.set(style="darkgrid")
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


# Definimos la semilla para el generador de número aleatorios
np.random.seed(810603)

# Dividimos los datos aleatoriamente en 80% para entrenamiento y 20% para prueba 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=25)
# IMPORTANTE: Las muestras están estratificadas, i.e., la proporción de clientes retenidos y no-retenidos es la misma en ambos

# Chequeemos los resultados
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#1.0 Arboles Aleatorios
# Building  Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion = 'entropy', random_state = 90)
rfc.fit(X_train, y_train)

# Evaluating on Training set
rfc_pred_train = rfc.predict(X_train)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 30)
dt.fit(X_train, y_train)
dt_pred_train = dt.predict(X_train)

feature_importance=pd.DataFrame({
    'rfc':rfc.feature_importances_,
    'dt':dt.feature_importances_
},index=df100.drop(columns=['transac']).columns)
feature_importance.sort_values(by='rfc',ascending=True,inplace=True)

index = np.arange(len(feature_importance))
fig, ax = plt.subplots(figsize=(18,38))
rfc_feature=ax.barh(index,feature_importance['rfc'],0.1,color='purple',label='Random Forest')
dt_feature=ax.barh(index+0.4,feature_importance['dt'],0.1,color='lightgreen',label='Decision Tree')
ax.set(yticks=index+0.4,yticklabels=feature_importance.index)

ax.legend()
plt.show()

columns_to_select = list(range(11, 15)) + [2, 6, 18, 19, 20, 21, 23, 25, 27, 31, 32]

# Creamos el nuevo DataFrame df200
Bases_Unit_SVM = df100.iloc[:, columns_to_select]
y1 = Bases_Unit_SVM['transac']
X1 = Bases_Unit_SVM.drop(columns = 'transac')

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 101)
X_oversample, y_oversample = smote.fit_resample(X_train, y_train)
X_train = X_oversample
y_train = y_oversample

# Metricas de los modelos

# curva ROC
def plot_roc(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1, drop_intermediate = False)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.001, 1.001])
    plt.ylim([-0.001, 1.001])
    plt.xlabel('1-Specificity (False Negative Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

# Matriz de confusión en el formato: cm[0,0], cm[0,1], cm[1,0], cm[1,1]: tn, fp, fn, tp

# Sensitivity
def custom_sensitivity_score(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return (tp/(tp+fn))

# Specificity
def custom_specificity_score(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return (tn/(tn+fp))

# Positive Predictive Value
def custom_ppv_score(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return (tp/(tp+fp))

# Negative Predictive Value
def custom_npv_score(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return (tn/(tn+fn))

# Accuracy
def custom_accuracy_score(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return ((tn+tp)/(tn+tp+fn+fp))

###Redes Neuronales
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
np.random.seed(77300)

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Definimos la arquitectura y métricas de optimización para la red

def crear_modelo():
    model = keras.Sequential([
        layers.Dense(128,  activation="relu", name="capa-1-oculta-128-neuronas"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu", name="capa-2-oculta-64-neuronas"),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid', name="capa-salida"),
    ])
    adam = tf.keras.optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

nn_estimators = []
nn_estimators.append(('estandarizar', StandardScaler())) # estandarizar los datos
nn_estimators.append(('multilayer-perceptron', KerasClassifier(build_fn=crear_modelo, epochs=30, batch_size=128,
                                             validation_split=0.2))) # compilar el modelo

# Definimos el modelo de Red Neuronal en TensorFlow y lo llamamos classifier_TF_NN
Classifier_TF_NN = Pipeline(nn_estimators, verbose=False)

# Entrenamos el modelo classifier_SVM con los datos de entrenamiento
Classifier_TF_NN.fit(X_train, y_train)

# Usamos el modelo entrenado para predecir sobre los datos de prueba
y_pred_prob = Classifier_TF_NN.predict_proba(X_test)[:,1] # probabilidades
y_pred = Classifier_TF_NN.predict(X_test)#np.where(y_pred_prob > class_threshold, 1, 0) # clasificación

# Revisamos las métricas del modelo
print('Métricas del modelo de Red Neuronal con Tensor Flow: \n')
cm = np.transpose(confusion_matrix(y_test, y_pred))
print("Matriz de confusión: \n" + str(cm))
print("                                   Accuracy: " + str(custom_accuracy_score(y_test, y_pred)))
print("                       SENSITIVITY (RECALL): " + str(custom_sensitivity_score(y_test, y_pred)))
print("                     SPECIFICITY (FALL-OUT): " + str(custom_specificity_score(y_test, y_pred)))
print("      POSITIVE PREDICTIVE VALUE (PRECISION): " + str(custom_ppv_score(y_test, y_pred)))
print("                  NEGATIVE PREDICTIVE VALUE: " + str(custom_npv_score(y_test, y_pred)))

plot_roc(y_test, y_pred_prob)
print(" AUC: " + str(roc_auc_score(y_test, y_pred_prob)))
