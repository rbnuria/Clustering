#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 17:24:32 2017

@author: nuria
"""



import pandas as pd
import sklearn.cluster as cl

from sklearn import preprocessing
from sklearn.neighbors import kneighbors_graph

import general as general

#CASO DE ESTUDIO 2: Queremos estudiar aquellos casos en los que se ha producido Colisión de vehículo en marcha,
#ya sea lateral, frontal, alndecance, etcétera, para estudiar si se realizan a una cierta hora o día de la semana.
#o alguna relación entre el número de muertos y otras variables etcétera.

#Si consideramos solo la clasificacion por tipo_accidente obtenemos 49280 muestras. Para disminuir la dimensión de la muestra,
#vamos a considerar el mismo problema en una sola comunidad autónoma, por ejemplo Andalucía.

#Definición del subcaso
def define_subset():
    accidentes = pd.read_csv('accidentes_2013.csv')
    

    subset = accidentes.loc[accidentes['TIPO_ACCIDENTE'].str.contains("Colisión de vehículos en marcha")]
    subset = subset.loc[subset['COMUNIDAD_AUTONOMA'].str.contains("Cataluña")]
    
    usadas = ['TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
    X = subset[usadas]

    return X


#Definimos los algoritmos a utilizar
def definition_clusters(subset):
    #Importante -> normalizar el conjunto de datos que utilizamos
    normalized_set = preprocessing.normalize(subset, norm='l2')

    print("-------- Definiendo los clusteres...")
    
    k_means = cl.KMeans(init='k-means++', n_clusters=5, n_init=100)
    
    # estimate bandwidth for mean shift
    bandwidth = cl.estimate_bandwidth(normalized_set, quantile=0.3)
    ms = cl.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    
    
    #Utilizarlo para casos de estudio pequeños
    spectral = cl.SpectralClustering(n_clusters=5, affinity="rbf")
    
    dbscan = cl.DBSCAN(eps=0.1)
    
    #Ponemos threshold bajo porque nos daba un warning en el fit_predict
    brc = cl.Birch(n_clusters = 5, threshold=0.1)
    
    #Los añadimos a una lista
    clustering_algorithms = (
        ('K-Means', k_means),
        ('MeanShift', ms),
        ('DBSCAN',dbscan),
        ('Birch', brc),
        ('SpectralClustering', spectral)
    )
    
    return clustering_algorithms


#Definimos el subset
subset2 = define_subset()
print("Tamaño del caso de estudio:" + str(len(subset2)))

#Definimos los algoritmos que vamos a utilizar
algorithms2 = definition_clusters(subset2)
predictions2 = general.get_predictions(algorithms2, subset2)

medidas2 = general.calculoMedidas(subset2, predictions2)



### Calculamos numero de clusteres para DBSCAN
k_dbscan = general.numero_clusters(predictions2[2])
k_meanshift = general.numero_clusters(predictions2[1])


############################## CONFIGURACIONES
##### DBSCAN

def configuraciones_dbscan():
    dbscan_01 = cl.DBSCAN(eps=0.01)
    dbscan_05 = cl.DBSCAN(eps=0.05)
    dbscan_1 = cl.DBSCAN(eps=0.1)
    
    #Los añadimos a una lista
    clustering_algorithms = (
        ('DBSCAN eps=0.01', dbscan_01),
        ('DBSCAN eps=0.05', dbscan_05),
        ('DBSCAN eps=0.1', dbscan_1)
    )
    
    return clustering_algorithms

config_dbscan = configuraciones_dbscan()
predictions_dbscan = general.get_predictions(config_dbscan, subset2)
medidas_dbscan = general.calculoMedidas(subset2, predictions_dbscan)

#Obtenemos cuantos clusters han constituido
k01_dbscan = general.numero_clusters(predictions_dbscan[0])
k05_dbscan = general.numero_clusters(predictions_dbscan[1])


### Birch
def configuraciones_birch():
    brc_1 = cl.Birch(n_clusters = 5, threshold=0.1)
    brc_05 = cl.Birch(n_clusters = 5, threshold=0.05)
    brc_01 = cl.Birch(n_clusters = 5, threshold=0.01)
    
    #Los añadimos a una lista
    clustering_algorithms = (
        ('Birch thershold=0.1', brc_1),
        ('Birch thershold=0.05', brc_05),
        ('Birch thershold=0.01', brc_01)
    )
    
    return clustering_algorithms    
        
        
        
config_birch = configuraciones_birch()
predictions_birch = general.get_predictions(config_birch, subset2)
medidas_birch = general.calculoMedidas(subset2, predictions_birch)


def configuraciones_birch2():
    brc_5 = cl.Birch(n_clusters = 5, threshold=0.1)
    brc_10 = cl.Birch(n_clusters = 10, threshold=0.1)
    brc_15 = cl.Birch(n_clusters = 15, threshold=0.1)
    brc_20 = cl.Birch(n_clusters = 20, threshold=0.1)
    brc_25 = cl.Birch(n_clusters = 25, threshold=0.1)
    brc_30 = cl.Birch(n_clusters = 30, threshold=0.1)
    
    #Los añadimos a una lista
    clustering_algorithms = (
        ('Birch-5', brc_5),
        ('Birch-10', brc_10),
        ('Birch-15', brc_15),
        ('Birch-20', brc_20),
        ('Birch-25', brc_25),
        ('Birch-30', brc_30)
    )
    
    return clustering_algorithms    
        
        
        
config_birch_2 = configuraciones_birch2()
predictions_birch_2 = general.get_predictions(config_birch_2, subset2)
medidas_birch_2 = general.calculoMedidas(subset2, predictions_birch_2)

        
        
### Kmeans
def configuraciones_kmeans():
    #normalized_set = preprocessing.normalize(subset, norm='l2')
    
    k_means_10 = cl.KMeans(init='k-means++', n_clusters=10, n_init=100)
    k_means_20 = cl.KMeans(init='k-means++', n_clusters=20, n_init=100)
    k_means_30 = cl.KMeans(init='k-means++', n_clusters=30, n_init=100)
    k_means_40 = cl.KMeans(init='k-means++', n_clusters=40, n_init=100)
    k_means_50 = cl.KMeans(init='k-means++', n_clusters=50, n_init=100)

        
    #Los añadimos a una lista
    clustering_algorithms = (
        ('K-Means-10', k_means_10),
        ('K-Means-20', k_means_20),
        ('K-Means-30', k_means_30),
        ('K-Means-40', k_means_40),
        ('K-Means-50', k_means_50)

    )
    
    return clustering_algorithms

config_kmeans = configuraciones_kmeans()
predictions_kmeans = general.get_predictions(config_kmeans, subset2)
medidas_kmeans = general.calculoMedidas(subset2, predictions_kmeans)
 
        
#scatter-matrix
general.scatter_matrix(predictions2[0], subset2)
general.scatter_matrix(predictions2[4], subset2)
        
        
        