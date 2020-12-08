# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:23:58 2020

@author: Mubtasim
"""

import numpy as np
import pandas as pd
# from scipy.linalg import svd

import matplotlib.pyplot as plt
import pylab as plt
import seaborn as sb
from IPython.display import Image
from IPython.core.display import HTML
from pylab import rcParams
# =============================================================================
# 
# import plotly.express as px
# 
# =============================================================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import scipy.linalg
from numpy.linalg import eig

import sklearn
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn import datasets

def main():
    dataset = r"C:\Users\Mubtasim\Desktop\Capstone\BUSPCA\Lyme PLOS - Machine Learning Database_10.16.2020.xlsx"
    excel = pd.read_excel(dataset)
    #import data into array

    data = pd.read_excel(dataset, na_values=['NA'], usecols = "D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R")
    print(data)
    print("\n")
    
# =============================================================================
#     data = excel.parse(0, usecols = range(3,18))
# =============================================================================
    X = data.to_numpy()
    variable_names = list(data.columns.values)
    print (variable_names)
    
    print(X.shape)
    print ("\n")
    
    #normalize for PCA
    Xnorm = normalize(X)

    # use SVD to find eigen values and vectors efficientally
    eigenvalues,eigenvectors = SVDforPCA(Xnorm)

    #define and find the eigen_vectors for dimension reduction
    alpha =.90
    refined_evectors = reduce_dimensions(eigenvalues,eigenvectors,alpha)
    k = len(refined_evectors)
    print("reduced to",k,"dimensions")
    print ("\n")
    
    #construct W matrix and reduce X to Y
    W = np.stack(refined_evectors, axis=1)
    Y=np.matmul(X,W)

    # make into dataframe
    df = pd.DataFrame(Y)
    
    print (df)
    print("\n")
    # save to xlsx file
    filepath = 'PCA.xlsx'
    try:
        df.to_excel(filepath, index=False)
    except:
        print("Didn't save. Please close",filepath)
        
    
#### new code below/////////////////////////////////    
    
    variable_names =  ['BBA65_Mean', 'BBA69_Mean', 'BBA70_Mean', 'BBA73_Mean', 'BmpA_Mean', 'DbpA_Mean', 'DbpB_Mean', 'ErpL_Mean', 'ErpY_Mean', 'OspC_Mean', 'P41_Mean', 'P45_Mean', 'P58_Mean', 'RevA_Mean', 'VlsE_Mean']
    
    pca = decomposition.PCA()
    data_pca = pca.fit_transform(X)
    pca.explained_variance_ratio_
   
    print(pca.explained_variance_ratio_.sum())
    myBasicCorr = df.corr()
   
    comps = pd.DataFrame(pca.components_ , columns = variable_names)
    sb.heatmap(myBasicCorr, annot = False, vmin = .75, vmax = 1, center = 0)
    sb.heatmap(comps)
    
# =============================================================================
#     df = px.data.data()
#     features = variable_names
#     
#     pca= PCA()
#     components = pca.fit_transform(df[features])
#     labels = {str(i): f"PC {i+1} ({var:.1f}%)"
#     for i, var in enumerate(pca.explained_variance_ratio_ * 100)
#     }
#     
#     fig = px.scatter_matrix(
#         components,
#         labels=labels,
#         dimensions=range(4),
#         color=df["species"]
#     )
#     fig.update_traces(diagonal_visible=False)
#     fig.show()
#         
# =============================================================================
    
    
    
#### old code below///////////////////////////////

    
    # normal to mean 0 and std 1 per column/expirement
def normalize(A):
    A_normalized = (A - np.mean(A, axis=0)) / np.std(A, axis=0)
    return A_normalized

# perform SVD in preparation for PCA
def SVDforPCA(A):
    #shape of A
    
    n, m = A.shape

    # Single Value Decomposition
    U, S, Vh = scipy.linalg.svd(A.T)

    eigen_vectors = U
    eigen_values = np.square(S)/(n-1)

    #normal eigenvalue and vector decomposition for PCA
    # eigen_values_eig, eigen_vectors_eig = np.linalg.eig(np.cov(A.T))
    # if(eigen_values == eigen_values_eig):
    #     print("eigen values are the same")
    return eigen_values,eigen_vectors
    print (eigen_values, " ", eigen_vectors)

def reduce_dimensions(evalue,evector,alpha):
#reduce eigenvectors until only alpha percent of the eigenvalue is represented by a new list
    evalue = np.asarray(evalue)
    evalue_total = np.sum(evalue)
    target = alpha*evalue_total

    refined_total = 0
    refined_evectors = []

    # attach evectors to evalues and sort from greatest to least
    for value,vector in sorted(zip(evalue,evector))[::-1]:
        refined_total+=value
        refined_evectors.append(vector)
        if(refined_total>=target):
            break

    #return list of the eigenvectors from reduction
    return refined_evectors



# ifmain 
if __name__ == '__main__':
    main()
