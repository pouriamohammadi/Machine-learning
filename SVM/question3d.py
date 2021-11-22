# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 15:30:12 2021

@author: Pouria
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance

def predict(df_train,df_test,kernel,c):
    error=0
    y_train = df_train['label'].to_numpy()
    y_train = np.reshape(y_train,(df_train.shape[0],1))
    for row in range(0, df_test.shape[0]):
        c_y_k= np.sum(np.multiply(np.multiply(y_train,c),np.reshape(kernel[:,row],(kernel.shape[0],1))))
        predict=np.sign(c_y_k)
        if predict==0:
            predict = 1
        y_test= df_test.iloc[row,-1]   
        if predict != y_test:
            error=error + 1
    return (error/df_test.shape[0])

#Main body
columns=['variance','skewness','curtosis','entropy','label']
#reading train
df_train=pd.read_csv('train.csv',names=columns,dtype=np.float64())
df_train=df_train.replace({'label':0},-1)
#reading test
df_test=pd.read_csv('test.csv',names=columns,dtype=np.float64())
df_test=df_test.replace({'label':0},-1)
# #Augmented matrices
df_train.insert(0,'b',1)
df_test.insert(0,'b',1)
#finding features and label
x_i = df_train.iloc[:,:-1].to_numpy()
y_i = df_train['label'].to_numpy()
y_i = np.reshape(y_i,(df_train.shape[0],1))
#hyperparameter
gamma_list=[ 0.1,.05,1,5,100]
epoch = 100
for gamma in gamma_list:
    #define kernel over x_i,x_i
    def func_kernel (u,v,gamma):
        k=np.zeros((u.shape[0],v.shape[0]))
        for i in range (u.shape[0]):
            for j in range (v.shape[0]):
                # dst = distance.euclidean(u[i,:],v[j,:])
                d = u[i,:] - v[j,:]
                dst = np.linalg.norm(d)
                k [i,j] = np.exp(-1*(dst**2)/gamma)
        return k
    kernel= func_kernel (x_i,x_i,gamma)
    c = np.zeros((df_train.shape[0],1))
    #Kernelize perceptron
    for T in range (0,epoch):
        for j in range (df_train.shape[0]):
            y_c= np.multiply(y_i,c)
            kernel_reshape= np.reshape(kernel[:,j],(kernel.shape[0],1))
            y_c_k_mat = np.multiply(kernel_reshape,y_c)
            y_c_k=np.sum(y_c_k_mat)
            # y_c_k = np.sum(np.multiply(np.multiply(y_i,c),np.reshape(kernel[:,j],(kernel.shape[0],1))))
            # y_c_k= 0
            # for i in range (df_train.shape[0]):
            #     y_c_k += y_i[i]*c[i]*kernel[i,j]
            y_pred = np.sign(y_c_k)
            if y_pred == 0:
                y_pred = 1
            #update c
            if y_pred != y_i[j]:
                c[j] = c[j]+1
    #Alpha final
    c_final=np.reshape(c,(df_train.shape[0],1))
    print('gamma=', gamma)
    
    #Error train
    train_error = predict(df_train,df_train,kernel,c_final)
    print ('train_error', train_error)
    #Error test
    x_i_test= df_test.iloc[:,:-1].to_numpy()
    kernel_test= func_kernel (x_i,x_i_test,gamma)
    test_error  = predict(df_train,df_test,kernel_test,c_final)
    print ('test_error', test_error,'\n')
    
