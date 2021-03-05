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

import scipy.linalg
from numpy.linalg import eig

import sklearn
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn import datasets

def main():
    dataset = r"C:\Users\Mubtasim\Desktop\Capstone\Lyme Data.xlsx"
    excel = pd.read_excel(dataset)
    #import data into array

    col = "D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R"
    data = pd.read_excel(dataset, na_values = 'NA', usecols = col)
    print(data)
    print("\n")
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
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
    alpha =.95
    refined_evectors = reduce_dimensions(eigenvalues,eigenvectors,alpha)
    k = len(refined_evectors)
    print("reduced to",k,"dimensions")
    print ("\n")
    print(eigenvalues)
    #construct W matrix and reduce X to Y
    W = np.stack(refined_evectors, axis=1)
    Y=np.matmul(X,W)

    # make into dataframe
    df = pd.DataFrame(Y)
    df = df.iloc[1:]
    
    print (df)
    print("\n")
    # save to xlsx file
    filepath = 'PCA_2.25.21.xlsx'
    try:
        df.to_excel(filepath, index=False)
    except:
        print("Didn't save. Please close",filepath)
        
    
#### new code below/////////////////////////////////    
    
    variable_names =  ['BBA65_Mean', 'BBA69_Mean', 'BBA70_Mean', 'BBA73_Mean', 'BmpA_Mean', 'DbpA_Mean', 'DbpB_Mean', 'ErpL_Mean', 'ErpY_Mean', 'OspC_Mean', 'P41_Mean', 'P45_Mean', 'P58_Mean', 'RevA_Mean', 'VlsE_Mean']
    
    pca = decomposition.PCA()
    data_pca = pca.fit_transform(X)
    data_new = pca.inverse_transform(data_pca)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.2)
    plt.scatter(data_new[:, 0], data_new[:, 1], alpha=0.8)
    plt.axis('equal');
    pca.explained_variance_ratio_
   
    print(pca.explained_variance_ratio_.sum())
    myBasicCorr = df.corr()
   
    comps = pd.DataFrame(pca.components_ , columns = variable_names)
    sb.heatmap(myBasicCorr, annot = False, vmin = .75, vmax = 1, center = 0)
    sb.heatmap(comps)
    
    print (pca.explained_variance_)
 # plot data
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, data):
    print (data)
    print(vector)
    v = (float(vector)) * (3.0) * (np.sqrt(length))
    draw_vector(pca.mean_, pca.mean_ + v)
    plt.axis('equal');    
 

# normal to mean 0 and std 1 per column/expi1rement
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
    
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
#    

# ifmain 
if __name__ == '__main__':
    main()
