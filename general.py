#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 17:31:44 2017

@author: nuria
"""

import time
import warnings
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn import preprocessing
from math import floor

np.seterr(divide='print', invalid='print')


#Obtención de las predicciones para cada algoritmo considerado
def get_predictions(clustering_algorithms, subset):

    #############En Clustering es muy importante la normalización
    normalized_set = preprocessing.normalize(subset, norm='l2')
    
    
    predictions = []
    
    print("------- Generando las predicciones...")
        
    for name, algorithm in clustering_algorithms:
        t0 = time.time()
        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            algorithm.fit(normalized_set)
        
        
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.fit_predict(normalized_set)
            
        t1 = time.time()

        predictions.append((name, y_pred, t1-t0))
        print(name)

                
    return predictions

#Print de la scatter_matrix
def scatter_matrix(prediction, subset):
    #Convertimos la asignación de clusteres a DataFrame
    clusters = pd.DataFrame(prediction[1], index = subset.index, columns = ['cluster'])
    
    nombre = prediction[0]
    
    #Añadimos como columna a X
    X_clusters = pd.concat([subset,clusters], axis = 1)
    
    print("------- Preprando el scatter matrix...")
    sns.set()
    variables = list(X_clusters)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_clusters, vars = variables, hue="cluster", palette = "Paired", plot_kws={"s":25}, diag_kind="hist")
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)
    sns_plot.savefig(nombre +".png")
    print("")
    
    
def calculoMedidas(subset, predictions):
    normalized_set = preprocessing.normalize(subset, norm='l2')
    
    meditions = []
    
    for pred in predictions:
        #CalculamosCalinski-Harabaz
        metric_CH = metrics.calinski_harabaz_score(normalized_set, pred[1])
        #Calculamos Silhouette en el 50% de la població,
        tric_SC = metrics.silhouette_score(normalized_set, pred[1], metric='euclidean', sample_size=floor(0.5*len(subset)), random_state=123456)

        meditions.append((pred[0],metric_CH, tric_SC))
        
    return meditions

def numero_clusters(prediction):
    y = prediction[1]
    y_labels = np.unique(y)
    y_labels = y_labels[y_labels>=0]
    
    return len(y_labels)

    
    
    