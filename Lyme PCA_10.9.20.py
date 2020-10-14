# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:04:55 2020

@author: Mubtasim
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
plt.show()


import pandas as pd

df = pd.read_excel(r'C:\Users\Mubtasim\Desktop\Capstone\Lyme PLOS - Machine Learning Database_10.9.2020.xlsx', 
                   names=['Sample','IgG WB Result','Disease Stage','BBA65 Mean','BBA69 Mean','BBA70 Mean',
                          'BBA73 Mean','BmpA Mean','DbpA Mean','DbpB Mean','ErpL Mean','ErpY Mean','OspC Mean',
                          'P41 Mean','P45 Mean','P58 Mean','RevA Mean','VlsE Mean'])
print (df)

from sklearn.preprocessing import StandardScaler
features = ['BBA65 Mean','BBA69 Mean','BBA70 Mean','BBA73 Mean','BmpA Mean','DbpA Mean','DbpB Mean','ErpL Mean',
            'ErpY Mean','OspC Mean','P41 Mean','P45 Mean','P58 Mean','RevA Mean','VlsE Mean']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['IgG WB Result']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['IgG WB Result']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

IgG_WB_Result = ['NEG','IT','POS',]
          
colors = ['r', 'g', 'b']
for IgG_WB_Result, color in zip(IgG_WB_Result,colors):
    indicesToKeep = finalDf['IgG WB Result'] == IgG_WB_Result
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(IgG_WB_Result)
ax.grid()

pca.explained_variance_ratio_

#once PCA 