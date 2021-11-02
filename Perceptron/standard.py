# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 11:25:48 2021

@author: Asus
"""

import pandas as pd
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt


def standard_perceptron(df, w, epoch, r):
    for e in range (0,epoch):
        # print('epoch',e)
        df_shuffle=df.sample(frac=1)
        # print(df_shuffle)
        df_shuffle=df_shuffle.reset_index(drop=True)
        # print(df_shuffle)
        for row in range (0,df_shuffle.shape[0]):
            y_i=df_shuffle.iloc[row,-1]
            # print('y_i',y_i)
            x_i=df_shuffle.iloc[row,:-1].to_numpy()
            # print('x_i',x_i)
            # print((2*y_i-1)*w.dot(x_i), '\n')
            if (2*y_i-1)*w.dot(x_i)<=0:
                w=w+(r*(2*y_i-1)*x_i.transpose())
                # print('w', w,'\n')
    return w            

def predict(df,w):
    error=0
    for row in range(0, df.shape[0]):
        x_i= df.iloc[row,:-1]        
        predict=np.sign(w.dot(x_i))
        if predict==0:
            predict =1
        if predict!=2*(df.iloc[row,-1])-1:
            error=error+1
    return (error/df.shape[0])

#Main body
columns=['variance','skewness','curtosis','entropy','label']

#reading train
df_train=pd.read_csv('train.csv',names=columns,dtype=np.float64())
#reading test
df_test=pd.read_csv('test.csv',names=columns,dtype=np.float64())
#J(w,b)
df_train.insert(0,'b',1)
df_test.insert(0,'b',1)


#hyperparameter
w_initial= np.zeros((1,df_train.shape[1]-1))
r=1
epoch=10

w_final=standard_perceptron(df_train, w_initial, epoch, r)
print ('w final', w_final)

error_test=predict(df_test,w_final)
print('error' , error_test)
    
