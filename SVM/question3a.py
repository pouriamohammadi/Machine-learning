# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 11:59:35 2021

@author: Asus
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize

def predict(df,w):
    error=0
    for row in range(0, df.shape[0]):
        predict = 0
        x_i= df.iloc[row,:-1]
        y_i= df.iloc[row,-1]    
        predict=np.sign(w.dot(x_i))
        if predict==0:
            predict =1
        if predict != y_i:
            error=error+1
    return (error/df.shape[0])

#Main body
columns=['variance','skewness','curtosis','entropy','label']
#reading train
df_train=pd.read_csv('train.csv',names=columns,dtype=np.float64())
df_train=df_train.replace({'label':0},-1)
#reading test
df_test=pd.read_csv('test.csv',names=columns,dtype=np.float64())
df_test=df_test.replace({'label':0},-1)
x_i = df_train.iloc[:,:-1].to_numpy()
y_i = df_train['label'].to_numpy()
y_i = np.reshape(y_i,(df_train.shape[0],1))

#hyperparameter
C = 700/873

# x_i_j=np.matmul(x_i,x_i.transpose())
# y_i_j=np.matmul(y_i,y_i.transpose())
# obj=np.multiply(x_i_j,y_i_j)
#function
def func(alpha):
    alpha = np.reshape(alpha,(df_train.shape[0],1))
    # alpha_i_j=np.matmul(alpha,alpha.transpose())
    # obj2=np.multiply(obj, alpha_i_j)
    #elementwise multipication
    xy_i=np.multiply(x_i,y_i)
    xyalpha_i=np.multiply(xy_i,alpha)
    xyalpha_j=xyalpha_i.transpose()
    #convert nested loops to matmul
    f= 0.5*(np.matmul(xyalpha_i,xyalpha_j).sum())-np.sum(alpha)
    # f= 0.5*(np.sum(obj2))-np.sum(alpha)
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

#w final
alpha_final=np.reshape(alpha_final,(df_train.shape[0],1))
xy_output=np.multiply(x_i,y_i)
w_out=np.multiply(xy_output,alpha_final).sum(axis=0)
print ('C= ', C)
print('w=',w_out)

#find list of b
w_t = w_out.reshape(1,w_out.shape[0])
b_list=[]
for j in range (alpha_final.shape[0]):
    if alpha_final[j]>0 and alpha_final[j]<C:
        x_j = x_i[j,:]
        b = y_i[j]- np.dot(w_t,x_j)
        b_list.append(b)
b=np.mean(b_list)
print('b=',b)

#Augmented matrices
w = np.insert(w_t,0,[b],axis=1)
df_train.insert(0,'b',1)
df_test.insert(0,'b',1)

train_error = predict(df_train,w)
test_error  = predict(df_test,w)

print ('train_error', train_error)
print ('test_error', test_error,'\n')


