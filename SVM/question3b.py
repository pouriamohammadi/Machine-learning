# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 21:47:57 2021

@author: Pouria
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize

def predict(df_train,df_test,b,kernel,alpha):
    error=0
    y_train = df_train['label'].to_numpy()
    y_train = np.reshape(y_train,(df_train.shape[0],1))
    for row in range(0, df_test.shape[0]):
        w_phi_xj=0
        for i in range (kernel.shape[0]):
            w_phi_xj += alpha_final[i]*y_train[i]*kernel[i,row]
        predict=np.sign(w_phi_xj+b)
        if predict==0:
            predict =1
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
#finding features and label
x_i = df_train.iloc[:,:-1].to_numpy()
y_i = df_train['label'].to_numpy()
y_i = np.reshape(y_i,(df_train.shape[0],1))
#hyperparameter
C = 500/873
gamma=100
#kernel function
def func_kernel (u,v):
    k=np.zeros((u.shape[0],v.shape[0]))
    for i in range (u.shape[0]):
        for j in range (v.shape[0]):
            d = u[i,:] - v[j,:]
            distance = np.linalg.norm(d)
            k [i,j] = np.exp(-(distance**2)/gamma)
    return k

#objective function
kernel= func_kernel (x_i,x_i)
y_i_j=np.matmul(y_i,y_i.transpose())
obj=np.multiply(kernel,y_i_j)
def func(alpha):
    alpha = np.reshape(alpha,(df_train.shape[0],1))
    alpha_i_j=np.matmul(alpha,alpha.transpose())
    obj2=np.multiply(obj, alpha_i_j)
    f= 0.5*(np.sum(obj2))-np.sum(alpha)
    return f
#constarint function
def constraint (alpha):
    alpha = np.reshape(alpha,(df_train.shape[0],1))
    return np.multiply(y_i,alpha).sum()
#constarint
cons = ({'type':'eq','fun':constraint })
#x0
x0= np.zeros((df_train.shape[0],1))
#bounds
bnds = [(0, C) for _ in x0] 
# bnds=[(0,C)]*df_train.shape[0]
#solver
solv = minimize (fun=func,x0=x0,method='SLSQP',bounds=bnds,constraints=cons)
alpha_final=solv.x

#W final and Alpha final
alpha_final=np.reshape(alpha_final,(df_train.shape[0],1))
print ('C= ', C)
print('gamma=', gamma)

#find list of b
b_list=[]
for j in range (alpha_final.shape[0]):
    if alpha_final[j]>0 and alpha_final[j]<C:
        w_phi_xj=0
        for i in range (alpha_final.shape[0]):
             w_phi_xj += alpha_final[i]*y_i[i]*kernel[i,j]
        b = y_i[j] - w_phi_xj
        b_list.append(b)
b=np.mean(b_list)
print('b=',b)

#Error
train_error = predict(df_train,df_train,b,kernel,alpha_final)
x_i_test= df_test.iloc[:,:-1].to_numpy()
kernel_test= func_kernel (x_i,x_i_test)
test_error  = predict(df_train,df_test,b,kernel_test,alpha_final)
print ('train_error', train_error)
print ('test_error', test_error,'\n')