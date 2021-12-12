# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:08:39 2021

@author: Pouria
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    sig= 1/(1+np.exp(-x))
    return sig

#Maximum a Postier        
def SVM_SGD(df,epoch,w_initial,Gamma_0,d,variance):
    objective_list=[]
    w = w_initial
    for t in range (0,epoch):
        Gamma_t=(Gamma_0/(1+(Gamma_0/d)*t))
        # print('Gamma_t',Gamma_t)
        #shuffle data
        df_shuffle = df.sample(frac=1)
        df_shuffle = df_shuffle.reset_index(drop=True)
        M = df_shuffle.shape[0]
        for row in range (0 , df_shuffle.shape[0]):
            x_i = np.reshape(df_shuffle.iloc[row,:-1].to_numpy(),(df.shape[1]-1,1))
            y_i = df_shuffle.iloc[row , -1]
            s= y_i*np.dot(w,x_i)
            #find derivative w for MAP
            derive_w = w/variance - M*y_i*(1-sigmoid(s))*(x_i.transpose())
            #update w
            w = w - Gamma_t * derive_w
        objective = loss(df_shuffle,w,variance)
        objective_list.append(objective)
    return w ,objective_list
            
def loss(df,w,variance):
    loss_sum=0
    for row in range(0, df.shape[0]):
        x_i= np.reshape(df.iloc[row,:-1].to_numpy(),(df.shape[1]-1,1))
        y_i= df.iloc[row,-1]    
        s = -y_i * np.dot(w,x_i)
        loss = np.log(1+np.exp(s))
        loss_sum = loss_sum + loss
    regulizer = (np.dot(w,w.transpose()))/(2*variance)
    obj = regulizer + loss_sum
    return obj[0][0]

def predict(df,w):
    error=0
    for row in range(0, df.shape[0]):
        predict = 0
        x_i= df.iloc[row,:-1]
        y_i= df.iloc[row,-1]    
        p1= w.dot(x_i)
        predict=np.sign(p1)
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
#J(b,w) 
df_train.insert(0,'b',1)
df_test.insert(0,'b',1)

#hyperparameter
print("input epoch value")
epoch=float(input())
print("input Gamma_0 value")
Gamma_0=float(input())
print("input d value")
d=float(input())
# Gamma_0=0.005
# d=0.01
# epoch=100
w_initial= np.zeros((1,df_train.shape[1]-1))
variance_list=[0.01,0.1,0.5,1,3,5,10,100]
# d_list=[100,10,1,0.1,0.01,0.001,0.0001]
# Gamma_0_list= [0.1,0.01,0.001,0.0001,0.00001]
for variance in variance_list:
    w_final , objective = SVM_SGD(df_train,epoch,w_initial,Gamma_0,d,variance)
    plt.plot(objective)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    title= "Gamma_0="+str(Gamma_0)+", d="+str(d)+", variance="+str(variance)
    plt.title(title)
    plt.show()
    train_error = predict(df_train,w_final)
    test_error = predict(df_test,w_final)
    print(title)
    print ('train_error', train_error)
    print ('test_error', test_error,'\n')


