# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:33:11 2021

@author: Pouria
"""
import pandas as pd
import numpy as np

def sigmoid(x):
    s = np.copy(x)
    for i in range (x.shape[0]):
        s [i,0] = 1/(1+np.exp(-x[i,0]))
    return s
def derivative_sigmoid(x):
    s = np.copy(x)
    for i in range (x.shape[0]):
        d= s[i,0]
        s[i,0] = d*(1-d)
    return s

class Network:
    def __init__(self, structure):
        self.structure=structure
        self.num_layers=len(structure)
        # self.B = [np.ones((l,1)) for l in structure[1:]]
        self.B = [np.array([[-1],[1]]),
                  np.array([[-1],[1]]),
                  np.array([-1])]
        # self.W = [np.ones((l,next_l)) for l,next_l in zip(structure[:-1], structure[1:])]
        self.W = [np.array([[-2,2],[-3,3]]),
                  np.array([[-2,2],[-3,3]]),
                  np.array([[2],[-1.5]])]
        
#hyperparameter
df_train= np.ones((2,1))
model = Network([2, 2, 2, 1])        
y_real = 1

#Forward propagation 
z_list=[]
a_list=[]
w=model.W
b=model.B
for i in range (model.num_layers-1):
    if i==0:
        z1=np.transpose(model.W[i])
        z2=df_train
        z3=np.matmul(np.transpose(model.W[i]),df_train)
        z4=model.B[i]
        z_i =z4+z3
        z_list.append(z_i)
        a_i = sigmoid(z_i)
        a_list.append(a_i)
    else:
        z1=np.transpose(model.W[i])
        z2=a_list[i-1]
        z3= np.matmul(np.transpose(model.W[i]),a_list[i-1])
        z4=model.B[i]
        z_i = z3+z4 
        z_list.append(z_i)
        a_i = sigmoid(z_i)
        a_list.append(a_i)
        if i == model.num_layers-2:
            a_list[i] = z_i
            y = z_i
#Backpropagation 
gamma = 1
delta_list=[]
deriv_w_list= []
deriv_b_list=[]
h_prime_list=[]
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
        delta = np.multiply(np.multiply(d1,d2),d3)
        delta_list.append(delta)           
        d4 =a_list[m-2]
        deriv_w = np.matmul(d4,delta.transpose()) 
        deriv_w_list.append(deriv_w)
        deriv_b_list.append(delta)
    else:    
        d1=delta_list[count-1]
        d2=model.W[m]
        d3=derivative_sigmoid(a_list[m-1])
        d4=np.matmul(d2,d1)
        delta = np.multiply(d4,d3)
        delta_list.append(delta)   
        deriv_b_list.append(delta)
        if m != 1 :
            d5=a_list[m-2]
            deriv_w = np.matmul(d5,delta.transpose()) 
            deriv_w_list.append(deriv_w)
        else:
            deriv_w = np.matmul(df_train,delta.transpose()) 
            deriv_w_list.append(deriv_w)
n = model.num_layers - 2
for layer in range(len(deriv_w_list)):
    m1 =model.W[n-layer] 
    m2 =np.multiply(deriv_w_list[layer],gamma)
    m3= m1-m2
    m4=model.B[n-layer]
    m5=np.multiply(deriv_b_list[layer],gamma)
    m6=m4-m5
    model.W[n-layer] = model.W[n-layer]-np.multiply(deriv_w_list[layer],gamma)
    model.B[n-layer] = model.B[n-layer]-np.multiply(deriv_b_list[layer],gamma)
    print("layer w =", n+1-layer ,"\n",deriv_w_list[layer])
    print("layer b =", n+1-layer ,"\n",deriv_b_list[layer],"\n")
w_final=model.W
b_final=model.B
