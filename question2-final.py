# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 18:58:06 2021

@author: Pouria
"""

import pandas as pd
import numpy as np
eps = np.finfo(float).eps
from numpy import log2 as log
from pprint import pprint

def entropy_H_GI(fraction,H_M_GI):
    entropy=0
    if H_M_GI == 'H':
        entropy+= -(fraction)*log(fraction+eps)
    elif H_M_GI == 'GI':
        entropy+= -(fraction)**2
    return entropy


def IG(df, att,H_M_GI):
    label_vars =df.label.unique()
    #print(att)
    #entropy of whole:
    entropy_s=0
    list_M_S=[]
    for labels in label_vars:
        num= len(df['label'][df.label== labels])
        den = len(df.label)
        frac_s=(num/(den+eps))
        #print(num,den)
        if H_M_GI != 'M':
            entropy_s += entropy_H_GI(frac_s,H_M_GI)
        else:
            list_M_S.append(frac_s)
    if H_M_GI == 'M':
        entropy_s = 1-max(list_M_S)
                   
    #print('entropy_s',entropy_s)
    #entropy of each attribute:
    features=df[att].unique()
    entropy=0
    #loop over features of each attribute including [low, high,vhigh ,...]
    for feat in features:
        entropy_att=0
        #print(feat)
        #loop over labels of each features including [acc, unacc,good ,...]
        for label_var in label_vars:
            if H_M_GI == 'M':
                list_M=[]
                #find majority:
                #second loop over labels [acc, unacc,good ,...] to find max
                for label_var2 in label_vars:
                    num = len(df[att][df[att]==feat][df.label ==label_var2])
                    den = len(df[att][df[att]==feat])             
                    frac = num/(den+eps)
                    list_M.append(frac)
                    #print(list_M)
                entropy_att = 1-max(list_M)
            else:
                #H or GI :
                num= len(df[att][df[att]==feat][df.label ==label_var])
                den = len(df[att][df[att]==feat])             
                frac = num/(den+eps)
                entropy_att += entropy_H_GI(frac,H_M_GI)
        #Gini Index:
        if H_M_GI == 'GI':
            entropy_att= 1-entropy_att
        #sum entropy of a feature
        frac_sigma = den/len(df)
        entropy += frac_sigma*entropy_att
        #print('entropy of feature', entropy,den,frac_sigma )
    I_G = entropy_s- entropy 
    return I_G
        
def choose_node (IG_dict):
    max_IG=0
    for attribute in IG_dict:
        if max_IG<IG_dict[attribute]:
            max_IG=IG_dict[attribute]
            choose_node=attribute
    return choose_node

def subtable (df,node,branch):
    df_sub=(df[df[node]==branch].reset_index(drop=True))
    return df_sub

              
def ID3_func(df,depth,main_tree=None,tree=None): 
    
    if depth == 0 and main_tree==None:
        common= df.label.mode()
        return common.get(0)
    
    attributes=[]
    attributes= df.keys()[:-1]
         
    if len(df.label.unique())==1 :
        return df.label[0] 
    
  #collecting IG of attributes
    else:
        IG_dict={}
        for att in attributes:
            inf_gain = IG(df, att, 'H')
            IG_dict[att]=inf_gain
        #finding the node 
        root_node=choose_node(IG_dict)
        
        #draw tree
        if tree is None:
            tree={}
            tree[root_node]={}
            
        #check the depth and main_tree  
        if depth == 0 and main_tree==True:
            common= df.label.mode()
            tree[root_node]=common.get(0)
            return tree
        
        depth=depth-1
        
        #finding subtable
        node_branches=[]
        node_branches=df[root_node].unique()
        for branch in node_branches:
            df_sub= subtable (df,root_node,branch)
            #print(branch)
            
            if len (df_sub) == 0:
                most_common= df.label.mode() 
                tree[root_node][branch]=most_common.get(0)
            else:
                tree[root_node][branch] =ID3_func(df_sub,depth)
            
    return tree

def predict(tree, df_row):           
    for key1 in tree.keys():
        #print (tree[key1])         
        if type(tree[key1]) is dict:
            for key2 in tree[key1].keys():
                tree_sub=tree[key1]
                if type(tree_sub[key2]) is not dict:
                    if df_row[key1]==key2:
                       predict_label= tree_sub[key2]
                       break
                else:
                    if df_row[key1]==key2:
                        tree_new=tree_sub[key2]   
                        label_pre= predict(tree_new, df_row)
                        predict_label=label_pre
                        break
        else:
            for key2 in tree[key1].keys():
                if df_row[key1]==tree[key1]:
                    predict_label= key2
                    break
    return predict_label

def test(df,tree):
    right=0
    for row in range (0,df.shape[0]):
         #print(row)
         df_rows = df.iloc[row , : ]
         #print(df_rows)
         pr_label= predict(tree,df_rows)
         
         #print(pr_label)
         if pr_label==df_rows.label:
             right+=1
    percent=right/df.shape[0]
    return  percent
        
                   
#Main body
columns=['buying','maint','doors','persons','lug_boot','safety','label']
data_train=[]    
tree={}
data_test=[] 

#train
train_file= open ( 'train.csv' , 'r' )
for line in train_file: 
    terms= line.strip().split(',')
    data_train.append (terms)
df_train= pd.DataFrame(data_train,columns=columns)
max_depth=4
tree = ID3_func(df_train,max_depth,True)
pprint(tree)

#test
test_file= open ( 'test.csv' , 'r' )
for line in test_file: 
    terms= line.strip().split(',')
    data_test.append (terms)
df_test= pd.DataFrame(data_test,columns=columns)
percent=test(df_test,tree)
print(percent)

#pprint(['safety', {'low': 'unacc', 'med': ['persons', {'2': 'unacc', '4': ['buying', {'high': ['lug_boot', {'med': ['doors', {'3': 'unacc', '4': 'acc', '2': 'unacc', '5more': ['maint', {'low': 'acc', 'vhigh': 'unacc', 'med': 'acc'}]}], 'small': 'unacc', 'big': 'acc'}], 'med': 'acc', 'low': 'acc', 'vhigh': 'unacc'}], 'more': 'unacc'}], 'high': 'unacc'}])



                    
                
            
        

   
#df.shape[0]=row
#df.iloc[[0]]['label'][0]

#den = len(df.label) 
#ss=[df['buying']=='low']
#var_att_list=var_att.tolist()
#x=df.label.unique()
#df[attribute][df[attribute]==variable][df.play ==target_variable]
#xxx=len(df.label) 
#for i in attributes:
 #   y=df[i].unique()
    #print (y)
#z=df.get('buying')
#xx=z.tolist()  
        



        
        
