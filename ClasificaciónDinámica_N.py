# -*- coding: utf-8 -*-
"""
Última actualización: 11 Oct 2023 02:34 PM
@autor: JULIAN FLOREZ
@colaborador: GUSTAVO BOBADILLA
"""

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
import seaborn as sns

# 1. Leer el archivo de Excel y tomar las columnas "Predios" y "Transacciones"
df = pd.read_excel("C:\\Users\\User\\OneDrive - Unidad de Planificación Rural Agropecuaria - UPRA\\3 Octubre\\Información\\Transacciones_N.xls")[["Predios", "Transacciones", "Categoria"]]

# Eliminar los registros donde la columna "Categoria" sea igual a "Ciudades y aglomeraciones"
#df = df[df['Categoria'] != 'Ciudades y aglomeraciones']
df = df[df['Transacciones'] > 0]

df1 = df
##########
# Determina el numero de clusters optimo
df['Categoria'].replace({
    'Ciudades y aglomeraciones': 0,
    'Intermedio': 1,
    'Rural': 2,
    'Rural disperso': 3
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
bars = plt.bar(range_n_clusters, silhouette_avg_metrics, color=colors)

# Añadir etiquetas de texto en la parte superior de cada barra
for bar, silhouette_avg in zip(bars, silhouette_avg_metrics):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(silhouette_avg, 2), va='bottom')  # va: vertical alignment

plt.xlabel('Number of Clusters')
plt.ylabel('Average Silhouette Score')
plt.title('Selection of Optimal Number of Clusters via Silhouette Method')
plt.colorbar(plt.cm.ScalarMappable(cmap="rainbow"))
plt.show()

# Use KMeans to cluster (or the number suggested by silhouette method)
kmeans_model = KMeans(n_clusters=5)
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
kmeans = KMeans(n_clusters=5, random_state=0).fit(df[['Transacciones']])
df['cluster'] = kmeans.labels_

# 3. Asignar etiquetas a los clusters basados en los centroides
sorted_idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
label_map = {
    sorted_idx[0]: '1. Muy Bajo',
    sorted_idx[1]: '2. Bajo',
    sorted_idx[2]: '3. Medio',
    sorted_idx[3]: '4. Alto',
    sorted_idx[4]: '5. Muy Alto'
}
df['clasificacion'] = df['cluster'].map(label_map)
df.drop('cluster', axis=1, inplace=True)

# 4. Visualizar la clasificación
colors = {'1. Muy Bajo':'green', '2. Bajo':'chartreuse', '3. Medio':'yellow', 
          '4. Alto':'orange', '5. Muy Alto':'red'}
plt.figure(figsize=(10, 6))

#chartreuse

# Dibuja cada punto de dato
for clasificacion, color in colors.items():
    subset = df[df['clasificacion'] == clasificacion]
    plt.scatter(subset['Predios'], subset['Transacciones'], s=50, c=color, label=str(clasificacion))

# Dibuja los centroides
plt.scatter(kmeans.cluster_centers_, kmeans.cluster_centers_, s=200, c='blue', marker='X', label='Centroides')

plt.title('Clasificación de Transacciones usando K-means')
plt.xlabel('Predios')
plt.ylabel('Transacciones')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# 5. Exportar los resultados a un nuevo archivo de Excel
path_to_save = "C:\\Users\\User\\OneDrive - Unidad de Planificación Rural Agropecuaria - UPRA\\3 Octubre\\Información\\Resultados_Transacciones_20231011.xlsx"
df.to_excel(path_to_save, index=False)

# Crear el DataFrame 'class' utilizando pivot_table para luego clasificar los resultados delos modelos
# de forma cualitativa segun las dinamicas
class_df = pd.pivot_table(df, values='Transacciones', index='clasificacion',
                          columns='Categoria', aggfunc='max')

# Renombrar las columnas
new_column_names = {
    0 : 'Ciudades y aglomeraciones',
    1 : 'Intermedio',
    2 : 'Rural',
    3 : 'Rural disperso'
}

class_df.rename(columns=new_column_names, inplace=True)