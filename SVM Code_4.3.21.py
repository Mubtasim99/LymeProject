# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:31:33 2021

@author: Mubtasim
"""

import numpy as np
import scipy
from scipy import optimize
    
path = r"C:\Users\Mubtasim\Desktop\Capstone\Lyme Data.csv"


df = np.genfromtxt(path, delimiter=',', skip_header=1, filling_values=-999, dtype='float', usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])


X=df[:, :-1]
Y=df[:, -1] 
#X=df[:,0:15]
#Y=df[:,16]

for i, x in enumerate(X):
    def func(w):
        return 0.5*np.sum(np.dot(w,w))
    
    def constraint1(w):
        zz= (Y[i]*np.dot(X[i],w))-1
        return zz
    
w0 = np.zeros(len(X[0]))

results = optimize.minimize(func, w0, constraints ={ "fun": constraint1, "type":"ineq"}, options={'disp':True})
print (results)

np.random.shuffle(df)
training, test = df[:100,:], df[100:,:]
print(training)
print(test)

XA=training[:, :-1]
YA=training[:, -1]
XB=test[:, :-1]
YB=test[:, -1]

for i, x in enumerate(XA):
    def func(w):
        return 0.5*np.sum(np.dot(w,w))
    
    def constraint1(w):
        zz= (YA[i]*np.dot(XA[i],w))-1
        return zz
    
w0 = np.zeros(len(XA[0]))

results = optimize.minimize(func, w0, constraints ={ "fun": constraint1, "type":"ineq"}, options={'disp':True})
print (results)

w2 = results.x

for i, x in enumerate (XB):
    #z3 is real values of Y
    z3 = (1-(np.dot(XB[i],w2)))
    
    if (z3 >=1.0):
    #z4 is the transformation of z3
        z4 =1.0
    elif (z3 <= -1.0):
        z4 =-1.0
        z5 = np.sum(z4-YB[i])/len(test)
        print ("The error value is ", z5*100)
#print("The error value is ", z5*100)