#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:42:43 2017

@author: nuria
"""



import pandas as pd
import sklearn.cluster as cl
import general as general



accidentes = pd.read_csv('accidentes_2013.csv')

subset = accidentes.sample(5000, random_state = 123456)
usadas = ['TOT_VICTIMAS', 'TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
subset = subset[usadas]

#Definimos el jerárquico y el DBSCAN con 0.3
#Definimos los algoritmos a utilizar
def definition_clusters(subset):
    
    k_means = cl.KMeans(init='k-means++', n_clusters=5, n_init=100)
    ward = cl.AgglomerativeClustering(n_clusters=100, linkage='ward')
    dbscan = cl.DBSCAN(eps=0.1)
        
    #Los añadimos a una lista
    clustering_algorithms = (
        ('K-Means', k_means),
        ('Ward', ward),
        ('DBSCAN', dbscan)
    )
    
    return clustering_algorithms

algoritmos = definition_clusters(subset)
predicciones = general.get_predictions(algoritmos, subset)

def delete_outliers_ward(prediction, subset):
    X = subset
    k = general.numero_clusters(prediction)
    #se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(prediction[1],index=X.index,columns=['cluster'])
    #y se añade como columna a X
    X_cluster = pd.concat([X, clusters], axis=1)
    
    #Filtro quitando los elementos (outliers) que caen en clusters muy pequeños en el jerárquico
    min_size = 2
    X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
    k_filtrado = len(set(X_filtrado['cluster']))
    #print('''De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos.
          #Del total de {:.0f} elementos, se seleccionan {:.0f}'''.format(k,k_filtrado,min_size,len(X),len(X_filtrado)))
    X_filtrado = X_filtrado.drop('cluster', 1)
    
    #X_filtrado_normal = preprocessing.normalize(X_filtrado, norm='l2')
    
    return X_filtrado


new_subset_ward = delete_outliers_ward(predicciones[1], subset)


def delete_outliers_dbscan(prediction, subset):
    labels = prediction[1]
    clusters_labels = pd.DataFrame(labels,index=subset.index,columns=['cluster'])
    cluster_subset = pd.concat([subset, clusters_labels], axis=1)
    new_subset = cluster_subset[clusters_labels['cluster']!=-1]

    return new_subset

new_subset_dbscan = delete_outliers_dbscan(predicciones[2], subset)

print(len(new_subset_ward))
print(len(new_subset_dbscan))

algorithms1 = definition_clusters(new_subset_ward)
algorithms2 = definition_clusters(new_subset_dbscan)

pred1 = general.get_predictions(algorithms1, new_subset_ward)
pred2 = general.get_predictions(algorithms2, new_subset_dbscan)
med1 = general.calculoMedidas(new_subset_ward, pred1)
med2 = general.calculoMedidas(new_subset_dbscan, pred2)

