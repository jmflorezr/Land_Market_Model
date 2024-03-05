# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 11:30:12 2023

@author: JULIAN FLOREZ
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_excel("C:\\Users\\JULIAN FLOREZ\\Downloads\\Indice01.xlsx")

# Aplicar logaritmo natural
data['CAMBIO_DE_PROPIETARIO_2016'] = np.log1p(data['CAMBIO_DE_PROPIETARIO_2016'])
data['Has._Coca_2016'] = np.log1p(data['Has._Coca_2016'])
data['POBLACION_CP_Y_RURAL_DISP_2016'] = np.log1p(data['POBLACION_CP_Y_RURAL_DISP_2016'])

# Aplicar KMeans
kmeans = KMeans(n_clusters=6, random_state=0)
data['CAMBIO_DE_PROPIETARIO_2016'] = kmeans.fit_predict(data[['CAMBIO_DE_PROPIETARIO_2016']])

# Convertir a categórica
data['CAMBIO_DE_PROPIETARIO_2016'] = data['CAMBIO_DE_PROPIETARIO_2016'].astype('category')

# Separar en características y etiquetas
X = data.drop('CAMBIO_DE_PROPIETARIO_2016', axis=1)
y = data['CAMBIO_DE_PROPIETARIO_2016']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Árbol de decisiones
dt_classifier = DecisionTreeClassifier(random_state=0)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

# Random Forest
rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

# Matriz de confusión y ROC-AUC para Árbol de Decisiones
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, dt_classifier.predict_proba(X_test), multi_class='ovr')

# Matriz de confusión y ROC-AUC para Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, rf_classifier.predict_proba(X_test), multi_class='ovr')
accuracy_dt = accuracy_score(y_test, y_pred_dt)
# Importancia de las características
feature_importance_dt = dt_classifier.feature_importances_
feature_importance_rf = rf_classifier.feature_importances_
accuracy_rf = accuracy_score(y_test, y_pred_rf)
# Resultados
print("Matriz de confusión (Árbol de Decisiones):", conf_matrix_dt)
print("ROC-AUC (Árbol de Decisiones):", roc_auc_dt)
print("Accuracy (Árbol de Decisiones):", accuracy_dt)
print("Importancia de las características (Árbol de Decisiones):", feature_importance_dt)
print("Matriz de confusión (Random Forest):", conf_matrix_rf)
print("ROC-AUC (Random Forest):", roc_auc_rf)
print("Accuracy (Random Forest):", accuracy_rf)
print("Importancia de las características (Random Forest):", feature_importance_rf)



##########################
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Cargar el archivo Excel
file_path = "C:\\Users\\JULIAN FLOREZ\\Downloads\\Indice01.xlsx"  # Cambia esto por la ruta de tu archivo
data = pd.read_excel(file_path)

# Aplicar logaritmo natural a las columnas especificadas
data['Ln_Has_Coca_2016'] = np.log1p(data['Has._Coca_2016'])
data['Ln_POBLACION_CP_Y_RURAL_DISP_2016'] = np.log1p(data['POBLACION_CP_Y_RURAL_DISP_2016'])

# Normalización min-max para todas las columnas
normalized_data_ln = (data - data.min()) / (data.max() - data.min())

# Aplicar el coeficiente de asimetría de Fisher a las columnas seleccionadas
skewness_ln_has_coca = skew(normalized_data_ln['Ln_Has_Coca_2016'])
skewness_ln_poblacion = skew(normalized_data_ln['Ln_POBLACION_CP_Y_RURAL_DISP_2016'])

# Crear nuevas columnas para los resultados en la tabla normalizada
normalized_data_ln['Skewness_Ln_Has_Coca_2016'] = skewness_ln_has_coca
normalized_data_ln['Skewness_Ln_POBLACION_CP_Y_RURAL_DISP_2016'] = skewness_ln_poblacion

# Crear la columna "Indice del mercado de tierras" utilizando la nueva fórmula
normalized_data_ln['Indice_del_mercado_de_tierras'] = ((
    normalized_data_ln['Ln_Has_Coca_2016'] * 0.04331697)+ 
    (normalized_data_ln['Ln_POBLACION_CP_Y_RURAL_DISP_2016'] * 0.25883212) +
    (normalized_data_ln['Actividades_Secundarias_2016'] * 0.26408716)+
    (normalized_data_ln['Participacion_Agregado_2016'] * 0.22075215)+
    (normalized_data_ln['Petracion_de_la_banda_ancha'] * 0.2130116)
)



# Exportar el dataframe a un archivo Excel
output_file_path = 'C:\\Users\\JULIAN FLOREZ\\Downloads\\Indice_Modificado.xlsx'  # Cambia esto por la ruta deseada
normalized_data_ln.to_excel(output_file_path, index=False)
