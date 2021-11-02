# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 11:25:48 2021

@author: Asus
"""

import pandas as pd
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

def voted_perceptron(df, w, epoch, r):
    c_m=0
    c_m_list=[]
    w_list=[]
    for e in range (0,epoch):
        df_shuffle=df.sample(frac=1)
        df_shuffle=df_shuffle.reset_index(drop=True)
        for row in range (0,df_shuffle.shape[0]):
            y_i=df_shuffle.iloc[row,-1]
            x_i=df_shuffle.iloc[row,:-1].to_numpy()
            # print('x_i',x_i)
            # print((2*y_i-1)*w.dot(x_i), '\n')
            if (2*y_i-1)*w.dot(x_i)<=0:
                c_m_list.append(c_m)
                w_list.append(w)
                w=w+(r*(2*y_i-1)*x_i.transpose())
                c_m=1
            else:
                c_m=c_m+1
    c_m_list.append(c_m)
    w_list.append(w)
    return c_m_list , w_list


def predict(df,c_m_list , w_list):
    error=0
    for row in range(0, df.shape[0]):
        predict_voted = 0
        x_i= df.iloc[row,:-1]
        for i in range (len(w_list)):
            w=w_list[i]
            predict=np.sign(w.dot(x_i))
            if predict==0:
                predict =1
            predict=predict*c_m_list[i]
            predict_voted=predict_voted+predict
            
        predict_voted=np.sign(predict_voted)
        if predict_voted==0:
            predict_voted =1    
        if predict_voted!=2*(df.iloc[row,-1])-1:
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

c_m_list=[]
w_list=[]
c_m_list , w_list =voted_perceptron(df_train, w_initial, epoch, r)
print ('w list', w_list)
print ('c_m_list', c_m_list)
print ('number of w is :', len(c_m_list))
# print ('number of w is :', len(w_list))

#output
# file_out = open("Output.txt","w")
# for i in range(0,len(c_m_list)):
#     file_out.write("Cm" +str(i) + ": "+str(c_m_list[i]) + "  W" +str(i)+": "+ str(w_list[i])+"\n")
# file_out.close()

error_test=predict(df_test,c_m_list , w_list)
print('error' , error_test)