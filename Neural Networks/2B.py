# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:33:11 2021

@author: Pouria
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    # s = x + b
    s = np.copy(x)
    for i in range (x.shape[0]):
        s [i,0] = 1/(1+np.exp(-x[i,0]))
    return s
def derivative_sigmoid(x):
    # s = x + b
    s = np.copy(x)
    for i in range (x.shape[0]):
        d= s[i,0]
        s[i,0] = d*(1-d)
    return s

class Network:
    def __init__(self, structure):
        self.structure=structure
        self.num_layers=len(structure)
        self.B = [np.random.randn(l,1) for l in structure[1:]]
        # self.B = [np.array([[-1],[1]]),
        #           np.array([[-1],[1]]),
        #           np.array([-1])]
        self.W = [np.random.randn(l,next_l) for l,next_l in zip(structure[:-1], structure[1:])]
        # self.W = [np.array([[-1,1],[-2,2],[-3,3]]),
        #           np.array([[-1,1],[-2,2],[-3,3]]),
        #           np.array([[-1],[2],[-1.5]])]
        # self.W = [np.array([[-2,2],[-3,3]]),
        #           np.array([[-2,2],[-3,3]]),
        #           np.array([[2],[-1.5]])]

def forward_propagation(model,df_train):
    z_list=[]
    a_list=[]
    for i in range (model.num_layers-1):
        if i==0:
            # df_train= np.reshape(np.insert(df_train,0,[1]),(3,1))
            z_i = np.matmul(np.transpose(model.W[i]),df_train)+model.B[i]
            z_list.append(z_i)
            a_i = sigmoid(z_i)
            a_list.append(a_i)
        else:
            # a_list[i-1]=np.reshape(np.insert(a_list[i-1],0,[1]),(3,1))
            z_i = np.matmul(np.transpose(model.W[i]),a_list[i-1])+ model.B[i]
            z_list.append(z_i)
            a_i = sigmoid(z_i)
            a_list.append(a_i)
            if i == model.num_layers-2:
                a_list[i] = z_i
                y = z_i
    return (y,a_list)

def Backpropagation(model,df_train,y_real):
    y,a_list = forward_propagation(model,df_train)
    delta_list=[]
    deriv_w_list= []
    deriv_b_list=[]
    count = -1
    for m in range (model.num_layers-1,0,-1):
        count+=1
        if m == model.num_layers-1:
            delta_y = y-y_real
            delta_list.append(delta_y)
            d1=a_list[m-2]
            d2=delta_y
            deriv_w = np.multiply(a_list[m-2],delta_y) 
            deriv_w_list.append(deriv_w)
            deriv_b_list.append(delta_y)
        elif m == model.num_layers-2:
            d1 =delta_list[count-1]
            d2 =model.W[m]
            d3 =derivative_sigmoid(a_list[m-1])
            delta = np.multiply(np.multiply(d2,d3),d1)
            delta_list.append(delta)           
            d4 =a_list[m-2]
            deriv_w = np.matmul(d4,delta.transpose()) 
            deriv_w_list.append(deriv_w)
            deriv_b_list.append(delta)
        elif  m != 1:    
            d1= delta_list[count-1]
            d2= model.W[m]
            d3= derivative_sigmoid(a_list[m-1])
            d4= np.multiply(d2,d3)
            delta = np.multiply(d4,d1)
            delta_list.append(delta)   
            deriv_b_list.append(delta)
            d5=a_list[m-2]
            deriv_w = np.matmul(d5,delta.transpose()) 
            deriv_w_list.append(deriv_w)
        else:
            d1=delta_list[count-1]
            d2=model.W[m]
            d3=derivative_sigmoid(a_list[m-1])
            d4=np.multiply(d3,d2)
            delta = np.matmul(d4,d1)
            delta_list.append(delta)   
            deriv_b_list.append(delta)
            deriv_w = np.matmul(df_train,delta.transpose()) 
            deriv_w_list.append(deriv_w)
    # for m in range (model.num_layers-1,0,-1):
    #     count+=1
    #     if m == model.num_layers-1:
    #         delta_y = y-y_real
    #         delta_list.append(delta_y)
    #         deriv_w = np.multiply(a_list[m-2],delta_y) 
    #         deriv_w_list.append(deriv_w)
    #         deriv_b_list.append(delta_y)
    #     elif m == model.num_layers-2:
    #         d1 =delta_list[count-1]
    #         d2 =model.W[m]
    #         d3 =derivative_sigmoid(a_list[m-1])
    #         delta = np.multiply(np.multiply(d1,d2),d3)
    #         delta_list.append(delta)           
    #         d4 =a_list[m-2].transpose()
    #         deriv_w = np.matmul(delta,d4) 
    #         deriv_w_list.append(deriv_w)
    #         deriv_b_list.append(delta)
    #     else:    
    #         d1=delta_list[count-1]
    #         d2=model.W[m-1]
    #         d3=derivative_sigmoid(a_list[m-1])
    #         delta = np.multiply(np.matmul(d2,d1),d3)
    #         delta_list.append(delta)   
    #         deriv_b_list.append(delta)
    #         if m != 1 :
    #             d4=a_list[m-2].transpose()
    #             deriv_w = np.matmul(delta,d4) 
    #             deriv_w_list.append(deriv_w)
    #         else:
    #             deriv_w = np.matmul(delta,df_train.transpose()) 
    #             deriv_w_list.append(deriv_w)
    return deriv_w_list , deriv_b_list


def SVM_SGD(df,model,epoch,Gamma_0,d):
    objective_list=[]
    for t in range (0,epoch):
        Gamma_t=(Gamma_0/(1+(Gamma_0/d)*t))
        # print('Gamma_t',Gamma_t)
        # w = w_initial
        #shuffle data
        df_shuffle = df.sample(frac=1)
        df_shuffle = df_shuffle.reset_index(drop=True)
        N = df_shuffle.shape[0]
        for row in range (0 , df_shuffle.shape[0]):
            x_i = np.reshape(df_shuffle.iloc[row,:-1].to_numpy(),(input_width,1))
            y_i = df_shuffle.iloc[row , -1]
            #updating sub-gradient
            deriv_w_list,deriv_b_list = Backpropagation(model,x_i,y_i)
            N=model.num_layers-2
            #update w and b
            for layer in range(len(deriv_w_list)):
                model.W[N-layer] = model.W[N-layer]-np.multiply(deriv_w_list[layer],Gamma_t)
                model.B[N-layer] = model.B[N-layer]-np.multiply(deriv_b_list[layer],Gamma_t)
            w_final=model.W
            b_final=model.B
        objective_list.append(loss(df,model))
    return w_final , b_final ,objective_list
    
def loss(df,model):
    loss_sum=0
    for row in range(0, df.shape[0]):
        predict = 0
        x_i= np.reshape(df.iloc[row,:-1].to_numpy(),(input_width,1))
        y_i= df.iloc[row,-1]    
        predict , a_list = forward_propagation(model,x_i)
        loss= 0.5*(predict[0][0]-y_i)**2 
        loss_sum= loss_sum + loss
    return loss_sum

def error(df,model):
    error=0
    for row in range(0, df.shape[0]):
        x_i= np.reshape(df.iloc[row,:-1].to_numpy(),(input_width,1))
        y_i= df.iloc[row,-1]    
        y_predict , a_list = forward_propagation(model,x_i)
        predict = np.sign(y_predict[0][0])
        if predict==0:
            predict =1
        if predict != (y_i):
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
#hyperparameter
print("input epoch value")
epoch=float(input())
print("input Gamma_0 value")
Gamma_0=float(input())
print("input d value")
d=float(input())

# epoch=50
# Gamma_0=0.05
# d=5
input_width=4
output_width=1
# hidden_width=20
width_list=[5,10,25,50,100]
# d_list=[100,10,1,0.1,0.01,0.001,0.0001]
# Gamma_0_list= [0.1,0.01,0.001,0.0001,0.00001]
for hidden_width in width_list:
    # for Gamma_0 in Gamma_0_list:
        # for d in d_list:
    model = Network([input_width,hidden_width-1,hidden_width-1,output_width])   
    w_initial=model.W     
    b_initial=model.B
    w , b, objective = SVM_SGD(df_train,model,epoch,Gamma_0,d)
    plt.plot(objective)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()
    train_error = error(df_train,model)
    test_error = error(df_test,model)
    print("epoch=",epoch,"Gamma_0=",Gamma_0,"d=",d,"hidden_width",hidden_width)
    print ('train_error', train_error)
    print ('test_error', test_error,'\n')

