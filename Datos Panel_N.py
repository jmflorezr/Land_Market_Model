# -*- coding: utf-8 -*-
"""
Última actualización: 11 Oct 2023 05:32 PM
@autor: JULIAN FLOREZ
@colaborador: GUSTAVO BOBADILLA
"""

import pandas as pd

###############Panel data

# 1. Leer el archivo de Excel tabla Maestra
# Ruta del archivo
ruta = "C:\\Users\\JULIAN FLOREZ\\Downloads\\Variables\\Copia de 20231116_DTR_TABLA_MAESTRA_INF_REGISTRAL.xlsx"
nombre_de_hoja = 'DP'
df1 = pd.read_excel(ruta, sheet_name=nombre_de_hoja, engine='openpyxl', header=0)

# Eliminar los registros donde la columna "Categoria" sea igual a "Ciudades y aglomeraciones"
#df11 = df1[df1['CategorIa_de_ruralidad'] != 'Ciudades y aglomeraciones']
#df11 = df1[df1['Categoria_de_ruralidad'] != 'Ciudades y aglomeraciones']
# Ojo, al eliminar datos de Ciudades reduce la capacidad de explicacion del modelo
# Iterar sobre las columnas 100 a 106 y aplicar la condición de filtrado
#for col in df1.columns[4:9]:
#    df1 = df1[df1[col] <= 1641]

#ruta = "C:\\Users\\User\\OneDrive - Unidad de Planificación Rural Agropecuaria - UPRA\\1 Agosto\\Modelos\\MunicipiosExcluidos.xlsx"
#nombre_de_hoja = 'MunicipiosExcluidos'
#MunicipiosExcluidos = pd.read_excel(ruta, sheet_name=nombre_de_hoja, engine='openpyxl', header=0)


# Obtén una lista de los valores de COD_MUN_EXCL en MunicipiosExcluidos
#codigos_excluidos = MunicipiosExcluidos['COD_MUN_EXCL'].tolist()
#mascara = ~df1['COD_MPIO'].isin(codigos_excluidos)
#df1 = df1[mascara]
    
df2 = df1.drop(columns=['DEPARTAMENTO', 'MUNICIPIO', 'CategorIa_de_ruralidad'])
df3 = df2.dropna(subset=['PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2019'])


# Para la variable dependiente
df_predios_cambio = pd.melt(df3, id_vars=['COD_MPIO'], 
                            value_vars=['PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2015', 'PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2016',
                                        'PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2017', 'PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2018', 'PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_2019'],
                            var_name='Year', value_name='Predios_Cambio')

df_predios_cambio['Year'] = df_predios_cambio['Year'].str.replace("PREDIOS__RURALES_CON_CAMBIO_DE_PROPIETARIO_", "")
df_predios_cambio['Year'] = df_predios_cambio['Year'].astype(int)

# Para el índice GINI
df_gini = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Gini_2015','Gini_2016','Gini_2017','Gini_2018','Gini_2019'], 
                  var_name='Year', value_name='Gini')
df_gini['Year'] = df_gini['Year'].str.replace("Gini_", "")
df_gini['Year'] = df_gini['Year'].astype(int)

# Para Predios Rurales
df_PredRur = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['PREDIOS_RURALES_2015','PREDIOS_RURALES_2016','PREDIOS_RURALES_2017','PREDIOS_RURALES_2018','PREDIOS_RURALES_2019'], 
                 var_name='Year', value_name='PRurales')
df_PredRur['Year'] =  df_PredRur['Year'].str.replace("PREDIOS_RURALES_", "")
df_PredRur['Year'] =  df_PredRur['Year'].astype(int)

# Para Hectáreas de Coca
df_Coca = pd.melt(df3, id_vars=['COD_MPIO'], 
                  value_vars=['Has._Coca_2015','Has._Coca_2016','Has._Coca_2017','Has._Coca_2018','Has._Coca_2019'], 
                  var_name='Year', value_name='Coca')
df_Coca['Year'] = df_Coca['Year'].str.replace("Has._Coca_", "", regex=True)
df_Coca['Year'] = df_Coca['Year'].astype(int)

# Para Permanentes
df_Perm = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Area_Permanentes_2015','Area_Permanentes_2016','Area_Permanentes_2017','Area_Permanentes_2018','Area_Permanentes_2019'], 
                  var_name='Year', value_name='Permanentes')
df_Perm['Year'] = df_Perm['Year'].str.replace("Area_Permanentes_", "")
df_Perm['Year'] = df_Perm['Year'].astype(int)

# Para Transitorios
df_Transit = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Area_Transitorios_2015','Area_Transitorios_2016','Area_Transitorios_2017','Area_Transitorios_2018','Area_Transitorios_2019'], 
                  var_name='Year', value_name='Transitorios')
df_Transit['Year'] = df_Transit['Year'].str.replace("Area_Transitorios_", "")
df_Transit['Year'] = df_Transit['Year'].astype(int)

# Para Cambio Permanentes
df_CPerm = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['CPerm%2015','CPerm%2016','CPerm%2017','CPerm%2018','CPerm%2019'], 
                  var_name='Year', value_name='CPerm')
df_CPerm['Year'] = df_CPerm['Year'].str.replace("CPerm%", "")
df_CPerm['Year'] = df_CPerm['Year'].astype(int)

# Para Cambio Transitorio
df_CTransi = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['CTransi2015','CTransi2016','CTransi2017','CTransi2018','CTransi2019'], 
                  var_name='Year', value_name='CTransi')
df_CTransi['Year'] = df_CTransi['Year'].str.replace("CTransi", "")
df_CTransi['Year'] = df_CTransi['Year'].astype(int)

# Para Hechos Victimizantes
df_vict = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Victimas_2015','Victimas_2016','Victimas_2017','Victimas_2018','Victimas_2019'], 
                  var_name='Year', value_name='Victimas')
df_vict['Year'] = df_vict['Year'].str.replace("Victimas_", "")
df_vict['Year'] = df_vict['Year'].astype(int)

# Para Indice de Riesgo de Victimización
df_IRV = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['IRV_2015','IRV_2016','IRV_2017','IRV_2018','IRV_2019'], 
                  var_name='Year', value_name='IRV')
df_IRV['Year'] = df_IRV['Year'].str.replace("IRV_", "")
df_IRV['Year'] = df_IRV['Year'].astype(int)

# Para Población Urbana
df_PoblCabecera = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Cabecera Municipal2015','Cabecera Municipal2016','Cabecera Municipal2017','Cabecera Municipal2018','Cabecera Municipal2019'], 
                  var_name='Year', value_name='PoblCabecera')
df_PoblCabecera['Year'] = df_PoblCabecera['Year'].str.replace("Cabecera Municipal", "")
df_PoblCabecera['Year'] = df_PoblCabecera['Year'].astype(int)

# Para Población Rural
df_poblrur = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['POBLACION_CP_Y_RURAL_DISP_2015','POBLACION_CP_Y_RURAL_DISP_2016','POBLACION_CP_Y_RURAL_DISP_2017','POBLACION_CP_Y_RURAL_DISP_2018','POBLACION_CP_Y_RURAL_DISP_2019'], 
                  var_name='Year', value_name='PoblRural')
df_poblrur['Year'] = df_poblrur['Year'].str.replace("POBLACION_CP_Y_RURAL_DISP_", "")
df_poblrur['Year'] = df_poblrur['Year'].astype(int)

# Para Actividad Primaria
df_act_primaria = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Actividades_Primarias_2015','Actividades_Primarias_2016','Actividades_Primarias_2017','Actividades_Primarias_2018','Actividades_Primarias_2019'], 
                  var_name='Year', value_name='act_primaria')
df_act_primaria['Year'] = df_act_primaria['Year'].str.replace("Actividades_Primarias_", "")
df_act_primaria['Year'] = df_act_primaria['Year'].astype(int)

# Para Actividad Secundaria
df_act_secundaria = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Actividades_Secundarias_2015','Actividades_Secundarias_2016','Actividades_Secundarias_2017','Actividades_Secundarias_2018','Actividades_Secundarias_2019'], 
                  var_name='Year', value_name='act_secundaria')
df_act_secundaria['Year'] = df_act_secundaria['Year'].str.replace("Actividades_Secundarias_", "")
df_act_secundaria['Year'] = df_act_secundaria['Year'].astype(int)

# Para Actividad Terciaria
df_act_terciaria = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Actividades_Terciarias_2015','Actividades_Terciarias_2016','Actividades_Terciarias_2017','Actividades_Terciarias_2018','Actividades_Terciarias_2019'], 
                  var_name='Year', value_name='act_terciaria')
df_act_terciaria['Year'] = df_act_terciaria['Year'].str.replace("Actividades_Terciarias_", "")
df_act_terciaria['Year'] = df_act_terciaria['Year'].astype(int)

# Para Valor Agregado
df_agregado = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Valor_Agregado_2015','Valor_Agregado_2016','Valor_Agregado_2017','Valor_Agregado_2018','Valor_Agregado_2019'], 
                  var_name='Year', value_name='agregado')
df_agregado['Year'] = df_agregado['Year'].str.replace("Valor_Agregado_", "")
df_agregado['Year'] = df_agregado['Year'].astype(int)

# Para Participacion Agregado
df_par_agreg = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Participacion_Agregado_2015','Participacion_Agregado_2016','Participacion_Agregado_2017','Participacion_Agregado_2018','Participacion_Agregado_2019'], 
                  var_name='Year', value_name='par_agreg')
df_par_agreg['Year'] = df_par_agreg['Year'].str.replace("Participacion_Agregado_", "")
df_par_agreg['Year'] = df_par_agreg['Year'].astype(int)

# Para POT
df_pot = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['Pot2015','Pot2016','Pot2017','Pot2018','Pot2019'], 
                  var_name='Year', value_name='pot')
df_pot['Year'] = df_pot['Year'].str.replace("Pot", "")
df_pot['Year'] = df_pot['Year'].astype(int)

# Para COLOCACIONES
df_coloc = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['COLOCACIONES_2015','COLOCACIONES_2016','COLOCACIONES_2017','COLOCACIONES_2018','COLOCACIONES_2019'], 
                  var_name='Year', value_name='COLOCACIONES')
df_coloc['Year'] = df_coloc['Year'].str.replace("COLOCACIONES_", "")
df_coloc['Year'] = df_coloc['Year'].astype(int)

# Para indice envejecimiento urbana
df_ind_env_urb = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['INDICE ENVEJECIMIENTO URBANA_2015','INDICE ENVEJECIMIENTO URBANA_2016','INDICE ENVEJECIMIENTO URBANA_2017','INDICE ENVEJECIMIENTO URBANA_2018','INDICE ENVEJECIMIENTO URBANA_2019'], 
                  var_name='Year', value_name='Ind_Envej_Urb')
df_ind_env_urb['Year'] = df_ind_env_urb['Year'].str.replace("INDICE ENVEJECIMIENTO URBANA_", "")
df_ind_env_urb['Year'] = df_ind_env_urb['Year'].astype(int)

# Para indice envejecimiento rural
df_ind_env_rur = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['INDICE ENVEJECIMIENTO RURAL_2015','INDICE ENVEJECIMIENTO RURAL_2016','INDICE ENVEJECIMIENTO RURAL_2017','INDICE ENVEJECIMIENTO RURAL_2018','INDICE ENVEJECIMIENTO RURAL_2019'], 
                  var_name='Year', value_name='Ind_Envej_Rur')
df_ind_env_rur['Year'] = df_ind_env_rur['Year'].str.replace("INDICE ENVEJECIMIENTO RURAL_", "")
df_ind_env_rur['Year'] = df_ind_env_rur['Year'].astype(int)

# Para Indice de Gobierno Abierto IGA
df_ind_gob_ab = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['IND_GOB_ABIERTO_2015','IND_GOB_ABIERTO_2016','IND_GOB_ABIERTO_2017','IND_GOB_ABIERTO_2018','IND_GOB_ABIERTO_2019'], 
                  var_name='Year', value_name='IND_GOB_ABIERTO')
df_ind_gob_ab['Year'] = df_ind_gob_ab['Year'].str.replace("IND_GOB_ABIERTO_", "")
df_ind_gob_ab['Year'] = df_ind_gob_ab['Year'].astype(int)

# Para Porcentaje de Varlos Agregado Actividades Primarias
df_porc_vr_agr_act_prim = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['PORC_VR_AGR_ACT_PRIM_2015','PORC_VR_AGR_ACT_PRIM_2016','PORC_VR_AGR_ACT_PRIM_2017','PORC_VR_AGR_ACT_PRIM_2018','PORC_VR_AGR_ACT_PRIM_2019'], 
                  var_name='Year', value_name='PORC_VR_AGR_ACT_PRIM')
df_porc_vr_agr_act_prim['Year'] = df_porc_vr_agr_act_prim['Year'].str.replace("PORC_VR_AGR_ACT_PRIM_", "")
df_porc_vr_agr_act_prim['Year'] = df_porc_vr_agr_act_prim['Year'].astype(int)

# Para Tasa de crecimiento de área de cultivos permanentes
df_TASA_CREC_AR_CULT_PERM = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['TASA_CREC_AR_CULT_PERM_2015','TASA_CREC_AR_CULT_PERM_2016','TASA_CREC_AR_CULT_PERM_2017','TASA_CREC_AR_CULT_PERM_2018','TASA_CREC_AR_CULT_PERM_2019'], 
                  var_name='Year', value_name='TASA_CREC_AR_CULT_PERM')
df_TASA_CREC_AR_CULT_PERM['Year'] = df_TASA_CREC_AR_CULT_PERM['Year'].str.replace("TASA_CREC_AR_CULT_PERM_", "")
df_TASA_CREC_AR_CULT_PERM['Year'] = df_TASA_CREC_AR_CULT_PERM['Year'].astype(int)

# Para Tasa de crecimiento de área de cultivos transitorios
df_TASA_CREC_AR_CULT_TRANSIT = pd.melt(df3, id_vars=['COD_MPIO'], value_vars=['TASA_CREC_AR_CULT_TRANSIT_2015','TASA_CREC_AR_CULT_TRANSIT_2016','TASA_CREC_AR_CULT_TRANSIT_2017','TASA_CREC_AR_CULT_TRANSIT_2018','TASA_CREC_AR_CULT_TRANSIT_2019'], 
                  var_name='Year', value_name='TASA_CREC_AR_CULT_TRANSIT')
df_TASA_CREC_AR_CULT_TRANSIT['Year'] = df_TASA_CREC_AR_CULT_TRANSIT['Year'].str.replace("TASA_CREC_AR_CULT_TRANSIT_", "")
df_TASA_CREC_AR_CULT_TRANSIT['Year'] = df_TASA_CREC_AR_CULT_TRANSIT['Year'].astype(int)

df_panel = (df_predios_cambio.merge(df_gini, on=['COD_MPIO', 'Year'])
                    .merge(df_PredRur, on=['COD_MPIO', 'Year'])
                    .merge(df_Coca, on=['COD_MPIO', 'Year'])
                    .merge(df_Perm, on=['COD_MPIO', 'Year'])
                    .merge(df_Transit, on=['COD_MPIO', 'Year'])
                    .merge(df_CPerm, on=['COD_MPIO', 'Year'])
                    .merge(df_CTransi, on=['COD_MPIO', 'Year'])
                    .merge(df_vict, on=['COD_MPIO', 'Year'])
                    .merge(df_IRV, on=['COD_MPIO', 'Year'])
                    .merge(df_PoblCabecera, on=['COD_MPIO', 'Year'])
                    .merge(df_poblrur, on=['COD_MPIO', 'Year'])
                    .merge(df_act_primaria, on=['COD_MPIO', 'Year'])
                    .merge(df_act_secundaria, on=['COD_MPIO', 'Year'])
                    .merge(df_act_terciaria, on=['COD_MPIO', 'Year'])
                    .merge(df_agregado, on=['COD_MPIO', 'Year'])
                    .merge(df_par_agreg, on=['COD_MPIO', 'Year'])
                    .merge(df_pot, on=['COD_MPIO', 'Year'])
                    .merge(df_coloc, on=['COD_MPIO', 'Year'])
                    .merge(df_ind_env_urb, on=['COD_MPIO', 'Year'])
                    .merge(df_ind_env_rur, on=['COD_MPIO', 'Year'])
                    .merge(df_ind_gob_ab, on=['COD_MPIO', 'Year'])
                    .merge(df_porc_vr_agr_act_prim, on=['COD_MPIO', 'Year'])
                    .merge(df_TASA_CREC_AR_CULT_PERM, on=['COD_MPIO', 'Year'])
                    .merge(df_TASA_CREC_AR_CULT_TRANSIT, on=['COD_MPIO', 'Year'])
                    )


df_panel = df_panel.sort_values(by=['COD_MPIO', 'Year'])
df_panel['Transitorios'] = df_panel['Transitorios'].fillna(0)
df_panel['pot'] = df_panel['pot'].fillna(0)
df_panel['Permanentes'] = df_panel['Permanentes'].fillna(0)
df_panel['CPerm'] = df_panel['CPerm'].fillna(0)
df_panel['CTransi'] = df_panel['CTransi'].fillna(0)
df_panel['Victimas'] = df_panel['Victimas'].fillna(0)
df_panel['IRV'] = df_panel['IRV'].fillna(0)
df_panel['TASA_CREC_AR_CULT_PERM'] = df_panel['TASA_CREC_AR_CULT_PERM'].fillna(0)
df_panel['TASA_CREC_AR_CULT_TRANSIT'] = df_panel['TASA_CREC_AR_CULT_TRANSIT'].fillna(0)
df_panel = df_panel.dropna(subset=["IND_GOB_ABIERTO"])
df_panel['PORC_VR_AGR_ACT_PRIM'] = df_panel['PORC_VR_AGR_ACT_PRIM'].fillna(0)
df_panel = df_panel.dropna(subset=["Gini"])



#Análisis descriptivo básico
print(df_panel[['Predios_Cambio', 'Gini', 'PRurales', 'Coca', 'Permanentes', 'Transitorios', 
                'CPerm', 'CTransi', 'Victimas', 'IRV', 'PoblCabecera', 'PoblRural',  
                'act_primaria', 'act_secundaria', 'act_terciaria', 'agregado', 'par_agreg', 
                'pot', 'COLOCACIONES', 'Ind_Envej_Urb', 'Ind_Envej_Rur',
                'IND_GOB_ABIERTO', 'PORC_VR_AGR_ACT_PRIM', 
                'TASA_CREC_AR_CULT_PERM', 'TASA_CREC_AR_CULT_TRANSIT']].describe())


#Visualizar tendencias a lo largo del tiempo:
import matplotlib.pyplot as plt

# Predios_Cambio tendencias
df_panel.groupby('Year')['Predios_Cambio'].mean().plot(label='Predios_Cambio', linestyle='--')

# Variables independientes
df_panel.groupby('Year')['Gini'].mean().plot(label='Gini', linestyle='--')
df_panel.groupby('Year')['PRurales'].mean().plot(label='PRurales', linestyle='--')
df_panel.groupby('Year')['Coca'].mean().plot(label='Coca', linestyle='--')
df_panel.groupby('Year')['Permanentes'].mean().plot(label='Permanentes', linestyle='--')
df_panel.groupby('Year')['Transitorios'].mean().plot(label='Transitorios', linestyle='--')
df_panel.groupby('Year')['CPerm'].mean().plot(label='CPerm', linestyle='--')
df_panel.groupby('Year')['CTransi'].mean().plot(label='CTransi', linestyle='--')
df_panel.groupby('Year')['Victimas'].mean().plot(label='Victimas', linestyle='--')
df_panel.groupby('Year')['IRV'].mean().plot(label='IRV', linestyle='--')
df_panel.groupby('Year')['PoblCabecera'].mean().plot(label='PoblCabecera', linestyle='--')
df_panel.groupby('Year')['PoblRural'].mean().plot(label='PoblRural', linestyle='--')
df_panel.groupby('Year')['act_primaria'].mean().plot(label='act_primaria', linestyle='--')
df_panel.groupby('Year')['act_secundaria'].mean().plot(label='act_secundaria', linestyle='--')
df_panel.groupby('Year')['act_terciaria'].mean().plot(label='act_terciaria', linestyle='--')
df_panel.groupby('Year')['agregado'].mean().plot(label='agregado', linestyle='--')
df_panel.groupby('Year')['par_agreg'].mean().plot(label='par_agreg', linestyle='--')
df_panel.groupby('Year')['pot'].mean().plot(label='pot', linestyle='--')
df_panel.groupby('Year')['COLOCACIONES'].mean().plot(label='COLOCACIONES', linestyle='--')
df_panel.groupby('Year')['Ind_Envej_Urb'].mean().plot(label='Ind_Envej_Urb', linestyle='--')
df_panel.groupby('Year')['Ind_Envej_Rur'].mean().plot(label='Ind_Envej_Rur', linestyle='--')
df_panel.groupby('Year')['IND_GOB_ABIERTO'].mean().plot(label='IND_GOB_ABIERTO', linestyle='--')
df_panel.groupby('Year')['PORC_VR_AGR_ACT_PRIM'].mean().plot(label='PORC_VR_AGR_ACT_PRIM', linestyle='--')
df_panel.groupby('Year')['TASA_CREC_AR_CULT_PERM'].mean().plot(label='TASA_CREC_AR_CULT_PERM', linestyle='--')
df_panel.groupby('Year')['TASA_CREC_AR_CULT_TRANSIT'].mean().plot(label='TASA_CREC_AR_CULT_TRANSIT', linestyle='--')


plt.legend()
plt.title('Tendencia de variables a lo largo del tiempo')
plt.show()


#Visualizar distribuciones
# Predios_Cambio
df_panel['Predios_Cambio'].hist(bins=30, alpha=0.5, label='Predios_Cambio')

# Variables independientes
df_panel['Gini'].hist(bins=30, alpha=0.5, label='Gini')
df_panel['PRurales'].hist(bins=30, alpha=0.5, label='PRurales')
df_panel['Coca'].hist(bins=30, alpha=0.5, label='Coca')
df_panel['Permanentes'].hist(bins=30, alpha=0.5, label='Permanentes')
df_panel['Transitorios'].hist(bins=30, alpha=0.5, label='Transitorios')
df_panel['CPerm'].hist(bins=30, alpha=0.5, label='CPerm')
df_panel['CTransi'].hist(bins=30, alpha=0.5, label='CTransi')
df_panel['Victimas'].hist(bins=30, alpha=0.5, label='Victimas')
df_panel['IRV'].hist(bins=30, alpha=0.5, label='IRV')
df_panel['PoblCabecera'].hist(bins=30, alpha=0.5, label='PoblCabecera')
df_panel['PoblRural'].hist(bins=30, alpha=0.5, label='PoblRural')
df_panel['act_primaria'].hist(bins=30, alpha=0.5, label='act_primaria')
df_panel['act_secundaria'].hist(bins=30, alpha=0.5, label='act_secundaria')
df_panel['act_terciaria'].hist(bins=30, alpha=0.5, label='act_terciaria')
df_panel['agregado'].hist(bins=30, alpha=0.5, label='agregado')
df_panel['par_agreg'].hist(bins=30, alpha=0.5, label='par_agreg')
df_panel['pot'].hist(bins=30, alpha=0.5, label='pot')
df_panel['COLOCACIONES'].hist(bins=30, alpha=0.5, label='COLOCACIONES')
df_panel['Ind_Envej_Urb'].hist(bins=30, alpha=0.5, label='Ind_Envej_Urb')
df_panel['Ind_Envej_Rur'].hist(bins=30, alpha=0.5, label='Ind_Envej_Rur')
df_panel['IND_GOB_ABIERTO'].hist(bins=30, alpha=0.5, label='IND_GOB_ABIERTO')
df_panel['PORC_VR_AGR_ACT_PRIM'].hist(bins=30, alpha=0.5, label='PORC_VR_AGR_ACT_PRIM')
df_panel['TASA_CREC_AR_CULT_PERM'].hist(bins=30, alpha=0.5, label='TASA_CREC_AR_CULT_PERM')
df_panel['TASA_CREC_AR_CULT_TRANSIT'].hist(bins=30, alpha=0.5, label='TASA_CREC_AR_CULT_TRANSIT')


plt.legend()
plt.title('Distribución de variables')
plt.show()

#Correlaciones:
correlations = df_panel[['Predios_Cambio', 'Gini', 'PRurales', 'Coca', 
                         'Permanentes', 'Transitorios', 'CPerm', 'CTransi',
                         'Victimas', 'IRV', 'PoblCabecera', 'PoblRural', 
                         'act_primaria', 'act_secundaria', 'act_terciaria', 
                         'agregado', 'par_agreg', 'pot',
                         'COLOCACIONES', 'Ind_Envej_Urb', 'Ind_Envej_Rur',
                         'IND_GOB_ABIERTO', 'PORC_VR_AGR_ACT_PRIM', 
                         'TASA_CREC_AR_CULT_PERM', 'TASA_CREC_AR_CULT_TRANSIT']].corr()
print(correlations)

#Analizar valores perdidos
missing_data = df_panel[['Predios_Cambio', 'Gini', 'PRurales', 'Coca', 
                         'Permanentes', 'Transitorios', 'CPerm', 'CTransi',
                         'Victimas', 'IRV', 'PoblCabecera', 'PoblRural', 
                         'act_primaria', 'act_secundaria', 'act_terciaria', 
                         'agregado', 'par_agreg', 'pot',
                         'COLOCACIONES', 'Ind_Envej_Urb', 'Ind_Envej_Rur',
                         'IND_GOB_ABIERTO', 'PORC_VR_AGR_ACT_PRIM', 
                         'TASA_CREC_AR_CULT_PERM', 'TASA_CREC_AR_CULT_TRANSIT']].isnull().sum()
print(missing_data)

#Modelo de datos de panel
from linearmodels.panel import PanelOLS

# Configura un multi-índice con COD_MPIO y Year
df_panel = df_panel.set_index(['COD_MPIO', 'Year'])

# Modelo de efectos fijos
formula = 'Predios_Cambio ~ 1 + Gini + PRurales + Coca + Permanentes + Transitorios + CPerm + CTransi + Victimas + IRV + PoblCabecera + PoblRural + act_primaria + act_secundaria + act_terciaria + par_agreg + pot + COLOCACIONES + Ind_Envej_Urb + Ind_Envej_Rur + IND_GOB_ABIERTO + PORC_VR_AGR_ACT_PRIM + TASA_CREC_AR_CULT_PERM + TASA_CREC_AR_CULT_TRANSIT + EntityEffects'
model = PanelOLS.from_formula(formula, data=df_panel)
results = model.fit()
print(results)


formula = 'Predios_Cambio ~ 1 + Gini + PRurales + Permanentes + IRV + PoblRural + act_primaria + act_secundaria + par_agreg + COLOCACIONES + Ind_Envej_Urb + EntityEffects'
model = PanelOLS.from_formula(formula, data=df_panel)
results = model.fit()
print(results)



###############################
##Modelo Inicial, Revivar R2 0.1621
###############################

#Modelo de efectos aleatorios
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# Preparación de los datos
df_panel = df_panel.reset_index() #Mirar cuando reset los indices- generar de nuevo el dataframe
dependent = df_panel['Predios_Cambio']
independent = df_panel[['Gini', 'PRurales', 'IRV', 'PoblCabecera',
                        'act_secundaria', 'act_terciaria', 'pot',
                        'COLOCACIONES']]
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
formula_fe = 'Predios_Cambio ~ 1 + Gini + PRurales + IRV + PoblCabecera + act_secundaria + act_terciaria + pot + COLOCACIONES + EntityEffects'
model_fe = PanelOLS.from_formula(formula_fe, data=df_panel)
results_fe = model_fe.fit()

# Efectos Aleatorios
formula_re = 'Predios_Cambio ~ 1 + Gini + PRurales + IRV + PoblCabecera + act_secundaria + act_terciaria + pot + COLOCACIONES'
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

