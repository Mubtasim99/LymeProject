# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 03:24:08 2021

@author: Mubtasim
"""

import lymePCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from SVM import SVM

dataset = r"C:\Users\Mubtasim\Desktop\Capstone\Lyme Data.xlsx"
excel = pd.read_excel(dataset)
    #import data into array

data = pd.read_excel(dataset, na_values=['NA'], skiprows = [0], usecols = "D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R")
print(data)
print("\n")
    
# =============================================================================
#     data = excel.parse(0, usecols = range(3,18))
# =============================================================================
X = data.to_numpy()
variable_names = list(data.columns.values)
print (variable_names)

y = variable_names

X, y =  data.make_blobs(n_samples=450, n_features=2, centers=2, cluster_std=1.05)
y = np.where(y == 0, -1, 1)

clf = SVM()
clf.fit(X, y)
#predictions = clf.predict(X)
 
print(clf.w, clf.b)

def visualize_svm():
     def get_hyperplane_value(x, w, b, offset):
          return (-w[0] * x + b + offset) / w[1]

     fig = plt.figure()
     ax = fig.add_subplot(1,1,1)
     plt.scatter(X[:,0], X[:,1], marker='o',c=y)

     x0_1 = np.amin(X[:,0])
     x0_2 = np.amax(X[:,0])

     x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
     x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

     x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
     x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

     x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
     x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

     ax.plot([x0_1, x0_2],[x1_1, x1_2], 'y--')
     ax.plot([x0_1, x0_2],[x1_1_m, x1_2_m], 'k')
     ax.plot([x0_1, x0_2],[x1_1_p, x1_2_p], 'k')

     x1_min = np.amin(X[:,1])
     x1_max = np.amax(X[:,1])
     ax.set_ylim([x1_min-3,x1_max+3])

     plt.show()

visualize_svm()
