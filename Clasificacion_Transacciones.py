# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:53:08 2023
@author: JULIAN FLOREZ
"""

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram

# 1. Leer el archivo de Excel y tomar las columnas "Predios" y "Transacciones"
df = pd.read_excel("C:\\Users\\JULIAN FLOREZ\\Downloads\\Variables\\Transacciones.xls")[["Predios", "Transacciones", "Categoria"]]

# Eliminar los registros donde la columna "Categoria" sea igual a "Ciudades y aglomeraciones"
df = df[df['Categoria'] != 'Ciudades y aglomeraciones']
df1 = df
##########
# Determina el numero de clusters optimo
df['Categoria'].replace({
    'Intermedio': 1,
    'Rural disperso': 2,
    'Rural': 3
}, inplace=True)

# Revisar Tipo de Datos existentes
data_types = df.dtypes

# Normalize the data
df_norm = scale(df)


# Estimate the optimal number of clusters using silhouette method
range_n_clusters = list(range(2,11))
silhouette_avg_metrics = []

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(df_norm)
    silhouette_avg = silhouette_score(df_norm, cluster_labels)
    silhouette_avg_metrics.append(silhouette_avg)

# Crear una paleta de colores 'rainbow'
colors = plt.cm.rainbow(np.linspace(0, 1, len(range_n_clusters)))

# Dibujo de las barras
plt.figure()
plt.bar(range_n_clusters, silhouette_avg_metrics, color=colors)
plt.xlabel('Number of Clusters')
plt.ylabel('Average Silhouette Score')
plt.title('Selection of Optimal Number of Clusters via Silhouette Method')
plt.colorbar(plt.cm.ScalarMappable(cmap="rainbow"))
plt.show()

# Use KMeans to cluster (or the number suggested by silhouette method)
kmeans_model = KMeans(n_clusters=6)
kmeans_model.fit(df_norm)
df['Cluster'] = kmeans_model.labels_

# Summary of clusters
summary_clusters = df.groupby('Cluster').mean()

# Prepare data for final plot
data_long = pd.melt(df, id_vars=['Cluster'], var_name='Feature', value_name='Value')

# Create the final plot
# Paleta personalizada: cambiar 'color1', 'color2', etc., por los colores que desees
custom_palette = sns.color_palette("husl", n_colors=data_long['Cluster'].nunique())

# Crear el gráfico final
sns.lineplot(data=data_long, x='Feature', y='Value', hue='Cluster', estimator=np.mean, err_style='bars', palette=custom_palette)

plt.show()
#################################

# 2. Usar k-means para clasificar las Transacciones
kmeans = KMeans(n_clusters=6, random_state=0).fit(df[['Transacciones']])
df['cluster'] = kmeans.labels_

# 3. Asignar etiquetas a los clusters basados en los centroides
sorted_idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
label_map = {
    sorted_idx[0]: 'muy bajo',
    sorted_idx[1]: 'bajo',
    sorted_idx[2]: 'medio bajo',
    sorted_idx[3]: 'medio',
    sorted_idx[4]: 'medio alto',
    sorted_idx[5]: 'alto'
}
df['clasificacion'] = df['cluster'].map(label_map)
df.drop('cluster', axis=1, inplace=True)

# 4. Visualizar la clasificación
colors = {'muy bajo':'blue', 'bajo':'green', 'medio bajo':'magenta', 'medio':'red', 'medio alto':'cyan', 'alto':'orange'}
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

# Crear el DataFrame 'class' utilizando pivot_table para luego clasificar los resultados delos modelos
# de forma cualitativa segun las dinamicas
class_df = pd.pivot_table(df, values='Transacciones', index='clasificacion',
                          columns='Categoria', aggfunc='max')

######################SOM
import numpy as np
import pandas as pd
from minisom import MiniSom 
import matplotlib.pyplot as plt

# # df es tu DataFrame
# df['Categoria'].replace({
#     'Ciudades y aglomeraciones': 0,
#     'Intermedio': 1,
#     'Rural disperso': 2,
#     'Rural': 3
# }, inplace=True)
# Realizado en las lineas anteriores

# Asegurándonos de que la columna "Categoria" es de tipo categórico
df['Categoria'] = df['Categoria'].astype('category')

data = df[['Predios', 'Transacciones', 'Categoria']].values

# Crear y entrenar el SOM
som = MiniSom(7, 7, 3, sigma=1.0, learning_rate=0.8)
som.train(data, 5000)

# Obtener las distancias para cada punto en el SOM
distances = np.array([som.distance_map()[som.winner(d)] for d in data])
labels = pd.cut(distances, 6, labels=['muy bajo', 'bajo', 'medio bajo', 'medio', 'medio alto','alto'])
df['clasificacion'] = labels

# Visualizar el SOM
weights = som.get_weights()

# Normalizar los pesos a [0,1]
normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())

plt.figure(figsize=(10, 10))
for i in range(normalized_weights.shape[0]):
    for j in range(normalized_weights.shape[1]):
        plt.plot(i, j, 'o', markerfacecolor=normalized_weights[i, j, :], markersize=10)

plt.title('SOM después de 5000 iteraciones')
plt.show()

# Gráfica cartesiana de 'Predios' vs 'Transacciones' coloreada por 'clasificacion'
colors = {'muy bajo':'blue', 'bajo':'green', 'medio bajo':'magenta', 'medio':'red', 'medio alto':'cyan', 'alto':'orange'}
plt.figure(figsize=(10, 10))
for label in df['clasificacion'].cat.categories:
    mask = df['clasificacion'] == label
    plt.scatter(df['Predios'][mask], df['Transacciones'][mask], c=colors[label], label=label)

plt.xlabel('Predios')
plt.ylabel('Transacciones')
plt.title('Gráfica de Predios vs Transacciones y segementada por clasificación ruralidad coloreada por clasificación')
plt.legend()
plt.show()

# Exportar los resultados a un nuevo archivo de Excel
path_to_save = "C:\\Users\\JULIAN FLOREZ\\Downloads\\Variables\\Resultados_Transacciones_SOM.xlsx"
df.to_excel(path_to_save, index=False)


###############Panel data

# 1. Leer el archivo de Excel tabla Maestra
# Ruta del archivo
ruta = "C:\\Users\\JULIAN FLOREZ\\Downloads\\Variables\\Tabla_Maestra.xlsx"

df1 = pd.read_excel(ruta, engine='openpyxl', header=0)
# Eliminar los registros donde la columna "Categoria" sea igual a "Ciudades y aglomeraciones"
#df1 = df1[df1['CategorIa_de_ruralidad'] != 'Ciudades y aglomeraciones']


df2 = df1.iloc[:, :106]
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

df_Coca = pd.melt(df3, id_vars=['COD_MPIO'], 
                  value_vars=['Has._Coca_2014','Has._Coca_2015','Has._Coca_2016','Has._Coca_2017','Has._Coca_2018','Has._Coca_2019'], 
                  var_name='Year', value_name='Coca')
df_Coca['Year'] = df_Coca['Year'].str.replace("Has._Coca_", "", regex=True)
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

df_act_primaria = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Actividades_Primarias_2014','Actividades_Primarias_2015','Actividades_Primarias_2016','Actividades_Primarias_2017','Actividades_Primarias_2018','Actividades_Primarias_2019'], 
                  var_name='Year', value_name='act_primaria')
df_act_primaria['Year'] = df_act_primaria['Year'].str.replace("Actividades_Primarias_", "")
df_act_primaria['Year'] = df_act_primaria['Year'].astype(int)


df_act_secundaria = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Actividades_Secundarias_2014','Actividades_Secundarias_2015','Actividades_Secundarias_2016','Actividades_Secundarias_2017','Actividades_Secundarias_2018','Actividades_Secundarias_2019'], 
                  var_name='Year', value_name='act_secundaria')
df_act_secundaria['Year'] = df_act_secundaria['Year'].str.replace("Actividades_Secundarias_", "")
df_act_secundaria['Year'] = df_act_secundaria['Year'].astype(int)

df_act_terciaria = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Actividades_Terciarias_2014','Actividades_Terciarias_2015','Actividades_Terciarias_2016','Actividades_Terciarias_2017','Actividades_Terciarias_2018','Actividades_Terciarias_2019'], 
                  var_name='Year', value_name='act_terciaria')
df_act_terciaria['Year'] = df_act_terciaria['Year'].str.replace("Actividades_Terciarias_", "")
df_act_terciaria['Year'] = df_act_terciaria['Year'].astype(int)

df_agregado = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Valor_Agregado_2014','Valor_Agregado_2015','Valor_Agregado_2016','Valor_Agregado_2017','Valor_Agregado_2018','Valor_Agregado_2019'], 
                  var_name='Year', value_name='agregado')
df_agregado['Year'] = df_agregado['Year'].str.replace("Valor_Agregado_", "")
df_agregado['Year'] = df_agregado['Year'].astype(int)

df_par_agreg = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Participacion_Agregado_2014','Participacion_Agregado_2015','Participacion_Agregado_2016','Participacion_Agregado_2017','Participacion_Agregado_2018','Participacion_Agregado_2019'], 
                  var_name='Year', value_name='par_agreg')
df_par_agreg['Year'] = df_par_agreg['Year'].str.replace("Participacion_Agregado_", "")
df_par_agreg['Year'] = df_par_agreg['Year'].astype(int)

df_pot = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Pot2014','Pot2015','Pot2016','Pot2017','Pot2018','Pot2019'], 
                  var_name='Year', value_name='pot')
df_pot['Year'] = df_pot['Year'].str.replace("Pot", "")
df_pot['Year'] = df_pot['Year'].astype(int)

df_PoblCabecera = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Cabecera Municipal2014','Cabecera Municipal2015','Cabecera Municipal2016','Cabecera Municipal2017','Cabecera Municipal2018','Cabecera Municipal2019'], 
                  var_name='Year', value_name='PoblCabecera')
df_PoblCabecera['Year'] = df_PoblCabecera['Year'].str.replace("Cabecera Municipal", "")
df_PoblCabecera['Year'] = df_PoblCabecera['Year'].astype(int)

df_CPerm = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['CPerm%2014','CPerm%2015','CPerm%2016','CPerm%2017','CPerm%2018','CPerm%2019'], 
                  var_name='Year', value_name='CPerm')
df_CPerm['Year'] = df_CPerm['Year'].str.replace("CPerm%", "")
df_CPerm['Year'] = df_CPerm['Year'].astype(int)

df_CTransi = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['CTransi2014','CTransi2015','CTransi2016','CTransi2017','CTransi2018','CTransi2019'], 
                  var_name='Year', value_name='CTransi')
df_CTransi['Year'] = df_CTransi['Year'].str.replace("CTransi", "")
df_CTransi['Year'] = df_CTransi['Year'].astype(int)





df_panel = (df_gini.merge(df_Coca, on=['COD_MPIO', 'Year'])
                    .merge(df_PredRur, on=['COD_MPIO', 'Year'])
                    .merge(df_Transit, on=['COD_MPIO', 'Year'])
                    .merge(df_Perm, on=['COD_MPIO', 'Year'])
                    .merge(df_vict, on=['COD_MPIO', 'Year'])
                    .merge(df_poblrur, on=['COD_MPIO', 'Year'])
                    .merge(df_act_primaria, on=['COD_MPIO', 'Year'])
                    .merge(df_act_secundaria, on=['COD_MPIO', 'Year'])
                    .merge(df_act_terciaria, on=['COD_MPIO', 'Year'])
                    .merge(df_agregado, on=['COD_MPIO', 'Year'])
                    .merge(df_par_agreg, on=['COD_MPIO', 'Year'])
                    .merge(df_pot, on=['COD_MPIO', 'Year'])
                    .merge(df_PoblCabecera, on=['COD_MPIO', 'Year'])
                    .merge(df_CPerm, on=['COD_MPIO', 'Year'])
                    .merge(df_CTransi, on=['COD_MPIO', 'Year'])
                    .merge(df_predios_cambio, on=['COD_MPIO', 'Year']))



df_panel = df_panel.sort_values(by=['COD_MPIO', 'Year'])
df_panel['Victimas'] = df_panel['Victimas'].fillna(0)
df_panel['CTransi'] = df_panel['CTransi'].fillna(0)
df_panel['CPerm'] = df_panel['CPerm'].fillna(0)
df_panel['Transitorios'] = df_panel['Transitorios'].fillna(0)

#Análisis descriptivo básico
print(df_panel[['Gini', 'Coca', 'PRurales', 'Transitorios', 'Permanentes', 'Victimas', 'PoblRural', 'act_primaria', 'act_secundaria', 'act_terciaria', 'agregado', 'par_agreg', 'pot', 'PoblCabecera', 'CPerm', 'CTransi', 'Predios_Cambio']].describe())


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
df_panel.groupby('Year')['act_primaria'].mean().plot(label='act_primaria', linestyle='--')
df_panel.groupby('Year')['act_secundaria'].mean().plot(label='act_secundaria', linestyle='--')
df_panel.groupby('Year')['act_terciaria'].mean().plot(label='act_terciaria', linestyle='--')
df_panel.groupby('Year')['agregado'].mean().plot(label='agregado', linestyle='--')
df_panel.groupby('Year')['par_agreg'].mean().plot(label='par_agreg', linestyle='--')
df_panel.groupby('Year')['pot'].mean().plot(label='par_agreg', linestyle='--')
df_panel.groupby('Year')['PoblCabecera'].mean().plot(label='par_agreg', linestyle='--')
df_panel.groupby('Year')['CPerm'].mean().plot(label='par_agreg', linestyle='--')
df_panel.groupby('Year')['CTransi'].mean().plot(label='par_agreg', linestyle='--')

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
df_panel['act_primaria'].hist(bins=30, alpha=0.5, label='act_primaria')
df_panel['act_secundaria'].hist(bins=30, alpha=0.5, label='act_secundaria')
df_panel['act_terciaria'].hist(bins=30, alpha=0.5, label='act_terciaria')
df_panel['agregado'].hist(bins=30, alpha=0.5, label='agregado')
df_panel['par_agreg'].hist(bins=30, alpha=0.5, label='par_agreg')

plt.legend()
plt.title('Distribución de variables')
plt.show()

#Correlaciones:
correlations = df_panel[['Predios_Cambio', 'Gini', 'Coca', 'PRurales', 'Transitorios', 'Permanentes', 'Victimas', 'PoblRural', 'act_primaria', 'act_secundaria', 'act_terciaria', 'agregado', 'par_agreg', 'pot', 'PoblCabecera', 'CPerm', 'CTransi']].corr()
print(correlations)

#Analizar valores perdidos
missing_data = df_panel[['Predios_Cambio', 'Gini', 'Coca', 'PRurales', 'Transitorios', 'Permanentes', 'Victimas', 'PoblRural', 'act_primaria', 'act_secundaria', 'act_terciaria', 'agregado', 'par_agreg', 'pot', 'PoblCabecera', 'CPerm', 'CTransi']].isnull().sum()
print(missing_data)

#Modelo de datos de panel
from linearmodels.panel import PanelOLS

# Configura un multi-índice con COD_MPIO y Year
df_panel = df_panel.set_index(['COD_MPIO', 'Year'])

# Modelo de efectos fijos
formula = 'Predios_Cambio ~ 1 + Gini + Coca + PRurales + Transitorios + Permanentes + Victimas + PoblRural + act_primaria + act_secundaria + act_terciaria + par_agreg + pot + PoblCabecera + CPerm + CTransi + EntityEffects'
model = PanelOLS.from_formula(formula, data=df_panel)
results = model.fit()
print(results)

formula = 'Predios_Cambio ~ 1 + Gini + PRurales + Victimas + act_secundaria + act_terciaria + pot + PoblCabecera + EntityEffects'
model = PanelOLS.from_formula(formula, data=df_panel)
results = model.fit()
print(results)
###############################
##Modelo Inicial, Revivar R2 0.1611
###############################

#Modelo de efectos aleatorios
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# Preparación de los datos
df_panel = df_panel.reset_index() #Mirar cuando reset los indices- generar de nuevo el dataframe
dependent = df_panel['Predios_Cambio']
independent = df_panel[['Gini', 'PRurales', 'Victimas', 'PoblCabecera']]
independent = sm.add_constant(independent)
df_panel.reset_index(drop=False, inplace=True)



  

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
#Modelo para el año 2016 sin serie de tiempo

# Definimos las columnas que queremos seleccionar
column_indices = [
    5, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 106, 
    113, 114, 115, 116, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 
    131, 132, 133, 134, 135, 139, 102
]


# Tomamos estas columnas del dataframe df1
df100 = df1.iloc[:, column_indices]

# # Eliminar las columnas especificadas del dataframe df100
# df101 = df100.drop(columns=["CategorIa_de_ruralidad"])

df101 = df100
# Eliminar las filas con valores nan en la columna especificada
df101 = df101.dropna(subset=["PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016"])

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

# Mostrar los resultados
print(model.summary())

#Ajustes modelo en variables

columns_to_select = [1, 2, 11, 16, 17, 38]

# Creamos el nuevo DataFrame df200
df200 = df101.iloc[:, columns_to_select]
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


# Definimos las columnas que queremos seleccionar
column_indices = [
    0, 5, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 106, 
    109, 111, 112, 113, 114, 115, 116, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 
    131, 132, 133, 134, 135, 139, 102
]
# Tomamos estas columnas del dataframe df1
df200 = df1.iloc[:, column_indices]
df200 = df200[df200['CategorIa_de_ruralidad'] != 'Ciudades y aglomeraciones']
df200.rename(columns={'CategorIa_de_ruralidad': 'Categoria_de_ruralidad'}, inplace=True)


df200 = df200.copy()  # Crear una copia explícita de df200
df200['Categoria_de_ruralidad'] = df200['Categoria_de_ruralidad'].astype('category')
#df200['CategorIa_de_municipio_2022'] = df200['CategorIa_de_municipio_2022'].astype('category')
df200['Sub_region'] = df200['Sub_region'].astype('category')
df200['Entorno_del_desarrollo'] = df200['Entorno_del_desarrollo'].astype('category')
df200.dropna(inplace=True)



# Crear la columna 'clasificacion' en df200
df200['clasificacion'] = None

# Ordenar class_df por la columna 'Intermedio' de mayor a menor
class_df = class_df.sort_values('Intermedio', ascending=False)

# Clasificar registros en df200
for col in ['Intermedio', 'Rural', 'Rural disperso']:  # Haciendo esto para cada columna que mencionaste
    sorted_class_df = class_df.sort_values(by=col, ascending=False)
    for idx, row in sorted_class_df.iterrows():
        mask = (df200['Categoria_de_ruralidad'] == col) & (df200['PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016'] <= row[col])
        df200.loc[mask, 'clasificacion'] = idx


# Definir un diccionario de mapeo para las clases
class_mapping = {
    'muy bajo': 0,
    'bajo': 1,
    'medio bajo': 2,
    'medio': 3,
    'medio alto': 4,
    'alto': 5
}

# Reemplazar las clases con los valores numéricos en la columna 'clasificacion'
df200['clase'] = df200['clasificacion'].replace(class_mapping)

df200.drop(['clasificacion', 'PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016'], axis=1, inplace=True)
df210 = df200
df200.drop(['COD_MPIO'], axis=1, inplace=True)


df200 = pd.get_dummies(df200, columns = df200.select_dtypes(exclude=['int32','int64','float64']).columns, drop_first = True)
pd.options.display.max_columns = None

df200.dropna(inplace=True)


y = df200['clase']
X = df200.drop(columns = 'clase')

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
},index=df200.drop(columns=['clase']).columns)
feature_importance.sort_values(by='rfc',ascending=True,inplace=True)

index = np.arange(len(feature_importance))
fig, ax = plt.subplots(figsize=(18,38))
rfc_feature=ax.barh(index,feature_importance['rfc'],0.1,color='purple',label='Random Forest')
dt_feature=ax.barh(index+0.4,feature_importance['dt'],0.1,color='lightgreen',label='Decision Tree')
ax.set(yticks=index+0.4,yticklabels=feature_importance.index)

ax.legend()
plt.show()

# Ordenar el DataFrame por la importancia de las características del modelo de Random Forest en orden descendente
feature_importance.sort_values(by='rfc', ascending=False, inplace=True)

# Seleccionar las 20 características más importantes
top_features = feature_importance.head(25)


# Convertir el índice del DataFrame top_20_features en una lista
top_features_list = top_features.index.tolist()

# Añadir la columna 'clase' a la lista de columnas a extraer
top_features_list.append('clase')

# Extraer las columnas relevantes del DataFrame df200 para crear df201
df201 = df200[top_features_list]

# Obtener los nombres de las columnas que tienen tipo de datos 'uint8'
uint8_columns = df201.select_dtypes(include=['uint8']).columns

# Cambiar el tipo de datos de estas columnas a 'int64'
for column in uint8_columns:
    df201[column] = df201[column].astype('int64')


# Contar la frecuencia de cada valor único en la columna 'clase'
value_counts = df201['clase'].value_counts()

# Mostrar las frecuencias
print(value_counts)




# Creamos el nuevo DataFrame df201
Bases_Unit_SVM = df201
y1 = Bases_Unit_SVM['clase']
X1 = Bases_Unit_SVM.drop(columns = 'clase')


# Dividimos los datos aleatoriamente en 80% para entrenamiento y 20% para prueba 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=25)
# IMPORTANTE: Las muestras están estratificadas, i.e., la proporción de clientes retenidos y no-retenidos es la misma en ambos



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

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier

def crear_modelo():
    model = keras.Sequential([
        layers.Dense(39,  activation="relu", name="capa-1-oculta-39-neuronas"),
        layers.Dropout(0.3),
        layers.Dense(15, activation="relu", name="capa-2-oculta-15-neuronas"),
        layers.Dropout(0.2),
        layers.Dense(6, activation='softmax', name="capa-salida"),  # 6 neuronas en la capa de salida, una para cada clase
    ])
    adam = tf.keras.optimizers.Adam()
    model.compile(loss='sparse_categorical_crossentropy',  # cambiar a categorical_crossentropy
                  optimizer=adam, 
                  metrics=['accuracy'])  # se mantiene accuracy como métrica
    return model

nn_estimators = []
nn_estimators.append(('estandarizar', StandardScaler()))  # estandarizar los datos
nn_estimators.append(('multilayer-perceptron', KerasClassifier(build_fn=crear_modelo, 
                                                               epochs=2000, 
                                                               batch_size=128, 
                                                               validation_split=0.2)))  # compilar el modelo

# Definimos el modelo de Red Neuronal en TensorFlow y lo llamamos Classifier_TF_NN
Classifier_TF_NN = Pipeline(nn_estimators, verbose=False)

# Entrenamos el modelo Classifier_TF_NN con los datos de entrenamiento
Classifier_TF_NN.fit(X_train, y_train)

# Usamos el modelo entrenado para predecir sobre los datos de prueba
y_pred_prob = Classifier_TF_NN.predict_proba(X_test)  # probabilidades
y_pred = Classifier_TF_NN.predict(X_test)  # clasificación



# Revisamos las métricas del modelo
print('Métricas del modelo de Red Neuronal con Tensor Flow: \n')
cm = np.transpose(confusion_matrix(y_test, y_pred))
print("Matriz de confusión: \n" + str(cm))
print("                                   Accuracy: " + str(custom_accuracy_score(y_test, y_pred)))
print("                       SENSITIVITY (RECALL): " + str(custom_sensitivity_score(y_test, y_pred)))
print("                     SPECIFICITY (FALL-OUT): " + str(custom_specificity_score(y_test, y_pred)))
print("      POSITIVE PREDICTIVE VALUE (PRECISION): " + str(custom_ppv_score(y_test, y_pred)))
print("                  NEGATIVE PREDICTIVE VALUE: " + str(custom_npv_score(y_test, y_pred)))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Crear matriz de confusión
cm = confusion_matrix(y_test, y_pred)  # Asumiendo que y_test contiene las etiquetas verdaderas

# Visualizar la matriz de confusión
sns.heatmap(cm, annot=True, fmt="d")
plt.ylabel('Verdaderos')
plt.xlabel('Predichos')
plt.show()


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# Binarizar las etiquetas (one-hot encoding)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5])
y_pred_prob_bin = label_binarize(y_pred, classes=[0, 1, 2, 3, 4, 5])

# Calcular la curva ROC y el área bajo la curva para cada clase
n_classes = 6  # Suponiendo 6 clases
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Trazar todas las curvas ROC
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='Curva ROC (área = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC para la clase {i}')
    plt.legend(loc="lower right")
    plt.show()




###################Precios
# 1. Leer el archivo de Excel y tomar las columnas "Predios" y "Transacciones"

ruta = "C:\\Users\\JULIAN FLOREZ\\Downloads\\Variables\\Precio_Merc_Final_V2.xlsx"
df1000 = pd.read_excel(ruta, engine='openpyxl', header=0)
# Lista de columnas a borrar
cols_to_drop_indices = [0, 1, 3, 9, 10, 11, 22, 23, 24, 27, 28, 29, 30, 37, 38, 39, 41, 45]

# Obtener los nombres de las columnas a partir de los índices
cols_to_drop = df1000.columns[cols_to_drop_indices]

# Borramos las columnas basadas en sus nombres
df1001 = df1000.drop(columns=cols_to_drop)

class_count = df1001["rango_precios"].value_counts()
class_count
