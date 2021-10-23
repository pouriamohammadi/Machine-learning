# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 17:54:18 2021

@author: Pouria
"""
import statistics
import pandas as pd
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt


def GD_train(df,r,norm_GD,w_t):
    i=0
    list_cost=[]
    while norm_GD>10**(-6):
        i+=1
        cost_function=0
        gradiant=np.zeros((1,w_t.shape[1]))
        for row in range (0, df.shape[0]):
            df_row=df.iloc[row,:-1].to_numpy()
            # print(df_train.loc[row,'output']-w.dot(df_row), '\n')
            cost_function+=(1/2)*(df.loc[row,'output']-w_t.dot(df_row))**2
            #update w
            gradiant=gradiant-(df.loc[row,'output']-w_t.dot(df_row))*df_row.transpose()
            # print(gradiant, '\n')
        w_t1=w_t-r*gradiant
        norm_GD=linalg.norm(w_t1-w_t,'fro')
        list_cost.append(cost_function)
        w_t=w_t1
        # print('i', i, norm_GD)
    print(w_t)
    print(w_t1)
    return list_cost, w_t1 , i

def GD_test(df,w):
    cost_function=0
    for row in range (0, df.shape[0]):
        df_row=df.iloc[row,:-1].to_numpy()
        cost_function+=(1/2)*(df.loc[row,'output']-w.dot(df_row))**2
    return cost_function

#Main body
columns=['Cement','Slag','Fly ash','Water','SP','Coarse Aggr','Fine Aggr','output']
# columns=['Cement','Slag','Fly ash','output']

#reading train
df_train=pd.read_csv('train.csv',names=columns,dtype=np.float64())
#reading test
df_test=pd.read_csv('train.csv',names=columns,dtype=np.float64())
#J(w,b)
df_train.insert(0,'b',1)
df_test.insert(0,'b',1)

#hyperparameter
w_initial= np.zeros((1,df_train.shape[1]-1))
r=0.014
norm_GD=1

list_cost , w_final, iteration=  GD_train(df_train,r,norm_GD,w_initial)
plt.title('cost function train')
plt.plot(range(1,iteration+1),list_cost)   
     
cost_function_test=GD_test(df_test,w_final)
print ('cost_function_test',cost_function_test)
