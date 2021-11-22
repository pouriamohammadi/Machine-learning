# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:08:39 2021

@author: Pouria
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt


def SVM_SGD(df,epoch,w_initial,Gamma_0,a,C):
    for t in range (0,epoch):
        Gamma_t=(Gamma_0/(1+(Gamma_0/a)*t))
        w = w_initial
        #shuffle data
        df_shuffle = df.sample(frac=1)
        df_shuffle = df_shuffle.reset_index(drop=True)
        N = df_shuffle.shape[0]
        for row in range (0 , df_shuffle.shape[0]):
            x_i = df_shuffle.iloc[row,:-1].to_numpy()
            y_i = df_shuffle.iloc[row , -1]
            #updating sub-gradient
            b = w[0,0]
            w_0 = np.delete(w,0,axis=1)
            z=(2*y_i-1)*w.dot(x_i)
            if (2*y_i-1)*w.dot(x_i)<=1:
                w_augmented = np.insert(w_0,0,[[0]],axis=1)
                x_i_t = x_i.transpose()
                w = w - Gamma_t*(w_augmented) + Gamma_t*C*N*(2*y_i-1)*(x_i.transpose())
            else:
                w0 = (1-Gamma_t)*w_0
                w= np.insert(w0,0,[[b]],axis=1)
    return w
            

def predict(df,w):
    error=0
    for row in range(0, df.shape[0]):
        predict = 0
        x_i= df.iloc[row,:-1]
        y_i= df.iloc[row,-1]    
        predict=np.sign(w.dot(x_i))
        if predict==0:
            predict =1
        if predict != 2*(y_i)-1:
            error=error+1
    return (error/df.shape[0])

#Main body
columns=['variance','skewness','curtosis','entropy','label']

#reading train
df_train=pd.read_csv('train.csv',names=columns,dtype=np.float64())
#reading test
df_test=pd.read_csv('test.csv',names=columns,dtype=np.float64())
#J(b,w) 
df_train.insert(0,'b',1)
df_test.insert(0,'b',1)

#hyperparameter
w_initial= np.zeros((1,df_train.shape[1]-1))
C_list=[100/873,500/873,700/873]
a_list=[100,10,1,0.1,0.01,0.001,0.0001]
Gamma_0_list= [0.1,0.01,0.001,0.0001,0.00001]
# file1 = open("results2a.txt","w")#append mode

for Gamma_0 in Gamma_0_list:
    for a in a_list:
        for C in C_list:
            epoch=100
            # Gamma_0=0.0001
            # a = 100
            # C = 100/873
            w = SVM_SGD(df_train,epoch,w_initial,Gamma_0,a,C)
            train_error = predict(df_train,w)
            test_error = predict(df_test,w)
            print ('C= ', C)
            # file1.write('\nC= '+ repr(C))
            print('Gamma_0=', Gamma_0 , ',a=' ,a )
            # file1.write('\nGamma_0='+ repr(Gamma_0) + ',a=' + repr(a))
            print ('w=', w)
            # file1.write('\nw='+ repr(w))
            print ('train_error', train_error)
            # file1.write('\ntrain_error'+ repr(train_error))
            print ('test_error', test_error,'\n')
            # file1.write('\ntest_error'+ repr(test_error)+ '\n')
# file1.close()