import numpy as np
import pandas as pd
# from scipy.linalg import svd
import scipy.linalg
from numpy.linalg import eig


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
