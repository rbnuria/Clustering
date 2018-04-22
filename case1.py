#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:38:37 2017

@author: nuria
"""


import pandas as pd
import sklearn.cluster as cl

from sklearn import preprocessing
from sklearn.neighbors import kneighbors_graph

import general as general

#CASO DE ESTUDIO 1: Queremos controlar el riesgo de la Barcelona a la hora de salida del colegio
#o dela salida del trabajo del turno de mañanas (1-4) en Barcelona

#Definición del subcaso
def define_subset():
    accidentes = pd.read_csv('accidentes_2013.csv')
    
    #Seleccionamos aquellos accidentes que hayan tenido lugar en la provincia de Barcelona
    subset = accidentes.loc[accidentes['PROVINCIA'].str.contains("Barcelona")]
    #Seleccionamos aquellos accidentes que hayan tenido lugar en la franjahoraria [13-16]
    subset = subset.loc[(subset['HORA'] >= 13) & (subset['HORA'] <= 16)]
    
    #subset = accidentes[accidentes['TIPO_ACCIDENTE'].str.contains("Colisión de vehículos")]
    #subset = subset.sample(1000, random_state=123456)

    
    #Seleccionamos aquellos atributos que queramos estudiar.    
    usadas = ['TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS']
    X = subset[usadas]

    return X
    

#Definimos los algoritmos a utilizar
def definition_clusters(subset):
    #Importante -> normalizar el conjunto de datos que utilizamos
    normalized_set = preprocessing.normalize(subset, norm='l2')

    print("-------- Definiendo los clusteres...")
    
    k_means = cl.KMeans(init='k-means++', n_clusters=5, n_init=100)
    
    two_means = cl.MiniBatchKMeans(n_clusters=5,  init='k-means++')

    
    # estimate bandwidth for mean shift
    bandwidth = cl.estimate_bandwidth(normalized_set, quantile=0.3)
    ms = cl.MeanShift(bandwidth=bandwidth)
    
    
    # connectivity matrix for structured Ward
    #connectivity = kneighbors_graph(normalized_set, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    #connectivity = 0.5 * (connectivity + connectivity.T)
    ward = cl.AgglomerativeClustering(n_clusters=100, linkage='ward')
    
    
    average = cl.AgglomerativeClustering(n_clusters=100, linkage='average')
    
    #Utilizarlo para casos de estudio pequeños
    #n_jobs = -1 para q vaya en paralelo
    #spectral = cl.SpectralClustering(n_clusters=3, affinity="nearest_neighbors",n_jobs=-1, n_neighbors = 3)
    
    #dbscan = cl.DBSCAN(eps=0.3)
        
    #Los añadimos a una lista
    clustering_algorithms = (
        ('K-Means', k_means),
        ('MeanShift', ms),
        ('MiniBatchMeans',two_means),
        ('AgglomerativeWard', ward),
        ('AgglomerativeAverage', average)
    )
    
    return clustering_algorithms

    

#Definimos el subset
subset1 = define_subset()
print("Tamaño del caso de estudio:" + str(len(subset1)))

#Definimos los algoritmos que vamos a utilizar
algorithms1 = definition_clusters(subset1)
predictions1 = general.get_predictions(algorithms1, subset1)

#Calculamos medidas de error
medidas1 = general.calculoMedidas(subset1, predictions1)

#Calculamos numero de clústeres para aquellos métodos a los que no se le especifique
n_clusters_meanshift = general.numero_clusters(predictions1[1])
print("El número de clústeres para MeanShift es {:.0f} ".format(n_clusters_meanshift))


########################## CONFIGURACIÓN DE PARAMETROS ##############################

############ kmeans

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
predictions_kmeans = general.get_predictions(config_kmeans, subset1)
medidas_kmeans = general.calculoMedidas(subset1, predictions_kmeans)

############ jerarquico

#Preprocesado -> eliminación de outliers

def delete_outliers(prediction, subset):
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
    print('''De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos.
          Del total de {:.0f} elementos, se seleccionan {:.0f}'''.format(k,k_filtrado,min_size,len(X),len(X_filtrado)))
    X_filtrado = X_filtrado.drop('cluster', 1)
    
    #X_filtrado_normal = preprocessing.normalize(X_filtrado, norm='l2')
    
    return X_filtrado
    
    


new_subset = delete_outliers(predictions1[3], subset1)


#Ejecutamos estos nuevos algoritmos.
pred_new_subset = general.get_predictions(algorithms1[3:5], new_subset)
medidas_new_subset = general.calculoMedidas(new_subset, pred_new_subset)

#Definimos algoritmo jerárquico con 100 clústeres:
ward_1000 = cl.AgglomerativeClustering(n_clusters=35, linkage='ward')

ward_1000_alg = (
        ('Ward-1000', ward_1000)
    )

pred_ward_1000 =  ('Ward-1000', ward_1000.fit_predict(new_subset))

new_subset_2 = delete_outliers(pred_ward_1000, pd.core.frame.DataFrame(new_subset))

#Ejecutamos estos nuevos algoritmos.

def configuraciones_jerarquicos():
    ward_5 = cl.AgglomerativeClustering(n_clusters=5, linkage='ward')
    average_5 = cl.AgglomerativeClustering(n_clusters=5, linkage='average')
    
    ward_10 = cl.AgglomerativeClustering(n_clusters=10, linkage='ward')
    average_10 = cl.AgglomerativeClustering(n_clusters=10, linkage='average')
                                            
    ward_15 = cl.AgglomerativeClustering(n_clusters=15, linkage='ward')
    average_15 = cl.AgglomerativeClustering(n_clusters=15, linkage='average') 

    ward_20 = cl.AgglomerativeClustering(n_clusters=20, linkage='ward')
    average_20 = cl.AgglomerativeClustering(n_clusters=20, linkage='average') 

    ward_25 = cl.AgglomerativeClustering(n_clusters=25, linkage='ward')
    average_25 = cl.AgglomerativeClustering(n_clusters=25, linkage='average')

    ward_30 = cl.AgglomerativeClustering(n_clusters=30, linkage='ward')
    average_30 = cl.AgglomerativeClustering(n_clusters=30, linkage='average')
    
    
        
    #Los añadimos a una lista
    clustering_algorithms = (
        ('Ward-5', ward_5),
        ('Average-5', average_5),
        ('Ward-10', ward_10),
        ('Average-10', average_10),
        ('Ward-15', ward_15),
        ('Average-15', average_15),
        ('Ward-20', ward_20),
        ('Average-20', average_20),
        ('Ward-25', ward_25),
        ('Average-25', average_25),
        ('Ward-30', ward_30),
        ('Average-30', average_30)

    )
    
    return clustering_algorithms


new_algoritmos = configuraciones_jerarquicos()  

pred_new_subset_2 = general.get_predictions(new_algoritmos, new_subset_2)
medidas_new_subset_2 = general.calculoMedidas(new_subset_2, pred_new_subset_2)


##################################### scatter-matrix
general.scatter_matrix(predictions1[0], subset1)


subset_dataframe = pd.core.frame.DataFrame(new_subset_2, columns=['TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS'])
general.scatter_matrix(pred_new_subset_2[0], subset_dataframe)

################################# dendograma
dendograma = new_subset_2.sample(500, random_state=123456)
dendograma_subset = preprocessing.normalize(dendograma, norm='l2')
dendograma_subsetDF = pd.DataFrame(dendograma_subset,index=dendograma.index,columns=['TOT_MUERTOS', 'TOT_HERIDOS_GRAVES', 'TOT_HERIDOS_LEVES', 'TOT_VEHICULOS_IMPLICADOS'])
sns.clustermap(dendograma_subsetDF, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)
