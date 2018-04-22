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

#CASO DE ESTUDIO 3: Queremos estudiar los accidentes que han ocurrido en condiciones buenas de conducción
# - LUMINOSIDAD: PLENO DÍA
# - FACTORES_ATMOSFÉRICOS: BUEN TIEMPO
# - SUPERFICIE_CALZADA: LIMPIA


#Definición del subcaso
def define_subset():
    accidentes = pd.read_csv('accidentes_2013.csv')
    

    subset = accidentes.loc[accidentes['LUMINOSIDAD'].str.contains("PLENO DÍA")]
    subset = subset.loc[subset['FACTORES_ATMOSFERICOS'].str.contains("BUEN TIEMPO")]
    subset = subset.loc[subset['SUPERFICIE_CALZADA'].str.contains("LIMPIA")]
    subset = subset.loc[subset['COMUNIDAD_AUTONOMA'].str.contains("Andalucía")]

    usadas = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
    X = subset[usadas]

    return X


#Definimos los algoritmos a utilizar EN ESTE CASO USAMOS LOS BASADOS EN DISTANCIAS DE PUNTOS (más comparables)
def definition_clusters(subset):
    #Importante -> normalizar el conjunto de datos que utilizamos
    normalized_set = preprocessing.normalize(subset, norm='l2')

    print("-------- Definiendo los clusteres...")
    
    k_means = cl.KMeans(init='k-means++', n_clusters=5, n_init=100)
    
    # estimate bandwidth for mean shift
    bandwidth = cl.estimate_bandwidth(normalized_set, quantile=0.3)
    ms = cl.MeanShift(bandwidth=bandwidth)
    
    two_means = cl.MiniBatchKMeans(n_clusters=5,  init='k-means++')
    
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(normalized_set, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    ward = cl.AgglomerativeClustering(n_clusters=5, linkage='ward')
    

    #dbscan = cl.DBSCAN(eps=0.3, n_clusters=5)
    
    brc = cl.Birch(n_clusters = 5, threshold=0.1)


        
    #Los añadimos a una lista
    clustering_algorithms = (
        ('K-Means', k_means),
        ('MiniBatchKMeans',two_means),
        ('MeanShift', ms),
        ('Agglomerative', ward),
        ('Birch', brc)
    )
    
    return clustering_algorithms


#Definimos el subset
subset3 = define_subset()
print("Tamaño del caso de estudio:" + str(len(subset3)))

#Definimos los algoritmos que vamos a utilizar
algorithms3 = definition_clusters(subset3)
predictions3 = general.get_predictions(algorithms3, subset3)


#general.scatter_matrix(predictions[2], subset)

medidas3 = general.calculoMedidas(subset3, predictions3)

k_meanshift = general.numero_clusters(predictions3[2])

######################### CAMBIO DE PARÁMETROS
##### Birch
def configuraciones_birch():
    brc_10 = cl.Birch(n_clusters = 10, threshold=0.01)
    brc_20 = cl.Birch(n_clusters = 25, threshold=0.01)
    brc_30 = cl.Birch(n_clusters = 30, threshold=0.01)
    brc_40 = cl.Birch(n_clusters = 40, threshold=0.01)
    brc_50 = cl.Birch(n_clusters = 50, threshold=0.01)
    brc_60 = cl.Birch(n_clusters = 60, threshold=0.01)

    #Los añadimos a una lista
    clustering_algorithms = (
        ('Birch-10', brc_10),
        ('Birch-20', brc_20),
        ('Birch-30', brc_30),
        ('Birch-40', brc_40),
        ('Birch-50', brc_50),
        ('Birch-60', brc_60)
    )
    
    return clustering_algorithms    
        
        
        
config_birch = configuraciones_birch()
predictions_birch = general.get_predictions(config_birch, subset3)
medidas_birch = general.calculoMedidas(subset3, predictions_birch)

def configuraciones_birch2():
    brc_01 = cl.Birch(n_clusters = 10, threshold=0.01)
    brc_05 = cl.Birch(n_clusters = 10, threshold=0.05)
    brc_07 = cl.Birch(n_clusters = 10, threshold=0.07)

    #Los añadimos a una lista
    clustering_algorithms = (
        ('Birch-01', brc_01),
        ('Birch-05', brc_05),
        ('Birch-07', brc_07),

    )
    
    return clustering_algorithms    
        
        
config_birch_2 = configuraciones_birch2()
predictions_birch_2 = general.get_predictions(config_birch_2, subset3)
medidas_birch_2 = general.calculoMedidas(subset3, predictions_birch_2)

#### Agglomerative
def configuraciones_agglomerative(subset):
    normalized_set = preprocessing.normalize(subset, norm = 'l2')
    
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(normalized_set, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    
    ward_10 = cl.AgglomerativeClustering(n_clusters=10, linkage='ward')
    ward_10_connectivity = cl.AgglomerativeClustering(n_clusters=10, linkage='ward', connectivity = connectivity)
    ward_20 = cl.AgglomerativeClustering(n_clusters=20, linkage='ward')
    ward_20_connectivity = cl.AgglomerativeClustering(n_clusters=20, linkage='ward', connectivity = connectivity)

    #Los añadimos a una lista
    clustering_algorithms = (
        ('Ward-10', ward_10),
        ('Ward-10-con', ward_10_connectivity),
        ('Ward-20', ward_20),
        ('Ward-20-con', ward_20_connectivity)
    )
    
    return clustering_algorithms    
        
        
        
config_ward = configuraciones_agglomerative(subset3)
predictions_ward = general.get_predictions(config_ward, subset3)
medidas_ward = general.calculoMedidas(subset3, predictions_ward)
  
#### KMEANS     
def configuraciones_minibatches(subset):    
    kmeans_10 = cl.KMeans(n_clusters=10,  init='k-means++', n_init = 100)
    kmeans_20 = cl.KMeans(n_clusters=20,  init='k-means++',n_init = 100)
    kmeans_30 = cl.KMeans(n_clusters=30,  init='k-means++',n_init = 100)
    kmeans_40 = cl.KMeans(n_clusters=40,  init='k-means++',n_init = 100)
    kmeans_50 = cl.KMeans(n_clusters=50,  init='k-means++',n_init = 100)
    kmeans_60 = cl.KMeans(n_clusters=60,  init='k-means++',n_init = 100)


    #Los añadimos a una lista
    clustering_algorithms = (
        ('Kmeans-10', kmeans_10),
        ('Kmeans-20', kmeans_20),
        ('Kmeans-30', kmeans_30),
        ('Kmeans-40', kmeans_40),
        ('Kmeans-50', kmeans_50),
        ('Kmeans-60', kmeans_60)
    )
    
    return clustering_algorithms    
        
        
        
config_batch = configuraciones_minibatches(subset3)
predictions_batch = general.get_predictions(config_batch, subset3)
medidas_batch = general.calculoMedidas(subset3, predictions_batch)

### MeanShift
def configuraciones_meanshift(subset): 
    normalized_set = preprocessing.normalize(subset, norm = 'l2')

    # estimate bandwidth for mean shift
    bandwidth1 = cl.estimate_bandwidth(normalized_set, quantile=0.3)
    bandwidth2 = cl.estimate_bandwidth(normalized_set, quantile=0.4)
    bandwidth3 = cl.estimate_bandwidth(normalized_set, quantile=0.5)


    ms1 = cl.MeanShift(bandwidth=bandwidth1)
    ms2 = cl.MeanShift(bandwidth=bandwidth2)
    ms3 = cl.MeanShift(bandwidth=bandwidth3)



    #Los añadimos a una lista
    clustering_algorithms = (
        ('MeanShift-1', ms1),
        ('MeanShift-2', ms2),
        ('MeanShift-3', ms3)
    )
    
    return clustering_algorithms    
        
        
        
config_meanshift = configuraciones_meanshift(subset3)
predictions_meanshift = general.get_predictions(config_meanshift, subset3)
medidas_meanshift = general.calculoMedidas(subset3, predictions_meanshift)

k_ms_1 = general.numero_clusters(predictions_meanshift[0])
k_ms_2 = general.numero_clusters(predictions_meanshift[1])
k_ms_3 = general.numero_clusters(predictions_meanshift[2])


### Scatter matrix
general.scatter_matrix(predictions3[0], subset3)
general.scatter_matrix(predictions3[3], subset3)

