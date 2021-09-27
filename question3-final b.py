# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 18:58:06 2021

@author: Pouria
"""
import statistics
import pandas as pd
import numpy as np
from numpy import log2 as log
from pprint import pprint
eps = np.finfo(float).eps

def entropy_H_GI(fraction, H_M_GI):
    
    if H_M_GI == 'H':
        entropy= -(fraction)*log(fraction+eps)
    elif H_M_GI == 'GI':
        entropy= -(fraction)**2
    return entropy


def IG(df, att, H_M_GI):
    label_vars =df.label.unique()

    #entropy of whole:
    entropy_s=0
    list_M_S=[]
    for labels in label_vars:
        num= len(df['label'][df.label== labels])
        den = len(df.label)
        frac_s=(num/(den+eps))
        #print(num,den)
        if H_M_GI != 'M':
            entropy_s += entropy_H_GI(frac_s, H_M_GI)
        else:
            list_M_S.append(frac_s)
    if H_M_GI == 'GI':
        entropy_s= 1+entropy_s
    if H_M_GI == 'M':
        entropy_s= 1-max(list_M_S)
                   

    #entropy of each attribute:
    features=df[att].unique()
    entropy_sigma=0
    #loop over features of each attribute including [low, high,vhigh ,...]
    for feat in features:
        entropy_att=0
        #print(feat)
        #loop over labels of each features including [acc, unacc,good ,...]
        for label_var in label_vars:
            # M
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
                
            # H or GI :
            else:
                num= len(df[att][df[att]==feat][df.label ==label_var])
                den = len(df[att][df[att]==feat])             
                frac = num/(den+eps)
                entropy_att += entropy_H_GI(frac,H_M_GI)
                
        #Gini Index:
        if H_M_GI == 'GI':
            entropy_att= 1+entropy_att
        #sum entropy of a feature
        frac_sigma = den/len(df)
        entropy_sigma += frac_sigma*entropy_att
        #print('entropy of feature', entropy,den,frac_sigma )
    InformationGain = entropy_s- entropy_sigma 
    return InformationGain
        
def choose_node (IG_dict):
    max_IG=0
    #print(IG_dict)
    for attribute in IG_dict:
        if max_IG<=IG_dict[attribute]:
            max_IG=IG_dict[attribute]
            choosen=attribute
        
    return choosen

def subtable (df,node,branch):
    df_sub=(df[df[node]==branch].reset_index(drop=True))
    return df_sub

              
def ID3_func(df,depth,H_M_GI,main_tree=None,tree=None): 
    
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
            inf_gain = IG(df, att, H_M_GI)
            IG_dict[att]=inf_gain
        #print(IG_dict)
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
                tree[root_node][branch] =ID3_func(df_sub,depth,H_M_GI)
            
    return tree

def predict(tree, df_row,max_depth):           
    for key1 in tree.keys():
        #print (tree[key1])         
        if type(tree[key1]) is dict:
            for key2 in tree[key1].keys():
                tree2=tree[key1]
                if type(tree2[key2]) is not dict:
                    predict_label= tree2[key2]
                    if df_row[key1]==key2:
                       predict_label= tree2[key2]
                       return predict_label
            
                else:
                    if df_row[key1]==key2:
                        tree_new=tree2[key2]   
                        label_pre= predict(tree_new, df_row,max_depth)
                        predict_label=label_pre
                        return predict_label
                        
            #make a guess since it is not in the tree           
            if predict_label is None:
                tree_new=tree2[key2]   
                label_pre= predict(tree_new, df_row,max_depth)
                predict_label=label_pre       
             
        else:
            if max_depth==0:
                predict_label= tree[key1]
                return predict_label

            else:
                for key2 in tree[key1].keys():
                    predict_label= key2
                    if df_row[key1]==tree[key1]:
                        predict_label= key2
                        return predict_label
                    
    
    return predict_label

def test(df,tree,max_depth):
    right=0
    for row in range (0,df.shape[0]):
         #print(row)
         df_rows = df.iloc[row , : ]
         #print(df_rows)
         pr_label= predict(tree,df_rows,max_depth)
         
         #print(pr_label)
         if pr_label==df_rows.label:
             right+=1
    percent=right/df.shape[0]
    return  percent

def data_modify(df):
    #attributes=[]
    #attributes= df.keys()[:-1]
    #print(attributes)
    med_list=[]
    for col in df.keys()[:-1]:
        if col== 'pdays':
            df= df.astype({'pdays': 'float'})
            df= df.replace({'pdays':-1.0}, 'unknown' )
            #med= df.loc[:,'pdays'].median(skipna=True)
            for row in range(0,df.shape[0]):
                if (df.loc[row,col])!='unknown':
                    med_list.append(df.loc[row,col])
            med=statistics.median(med_list)
            #print(col,med)
            for row in range(0,df.shape[0]):
                    if (df.loc[row,col])!='unknown':
                        if (df.loc[row,col])<med:
                            df.loc[row,col]=0
                        else:
                            df.loc[row,col]=1
        else:        
            #x=df.loc[:,col].unique()
            if df.loc[1,col].isdigit()== True:
                df=df.astype({col: 'float'})
                med= df.loc[:,col].median(skipna=True)
                #print(col,med)
                for row in range(0,df.shape[0]):
                    if (df.loc[row,col])<med:
                        df.loc[row,col]=0
                    else:
                        df.loc[row,col]=1
    return df

def data_modify_missing(df):
    #attributes=[]
    #attributes= df.keys()[:-1]
    #print(attributes)
    med_list=[]
    for col in df.keys()[:-1]:
        if col== 'pdays':
            df= df.astype({'pdays': 'float'})
            df= df.replace({'pdays':-1.0}, 'unknown' )
            #med= df.loc[:,'pdays'].median(skipna=True)
            for row in range(0,df.shape[0]):
                if (df.loc[row,col])!='unknown':
                    med_list.append(df.loc[row,col])
            med=statistics.median(med_list)
            mod=statistics.mode(med_list)
            #print(col,med)
            for row in range(0,df.shape[0]):
                    if (df.loc[row,col])!='unknown':
                        if (df.loc[row,col])<med:
                            df.loc[row,col]=0
                        else:
                            df.loc[row,col]=1
                    if (df.loc[row,col])=='unknown':
                        if mod<med:
                            df.loc[row,col]=0
                        else:
                            df.loc[row,col]=1
        else:        
            #x=df.loc[:,col].unique()
            if df.loc[1,col].isdigit()== True:
                df=df.astype({col: 'float'})
                med= df.loc[:,col].median(skipna=True)
                #print(col,med)
                for row in range(0,df.shape[0]):
                    if (df.loc[row,col])<med:
                        df.loc[row,col]=0
                    else:
                        df.loc[row,col]=1
            else:
                column=[]
                for row in range(0,df.shape[0]):
                    if df.loc[row,col] != 'unknown':
                        column.append(df.loc[row,col])
                mode_col=statistics.mode(column)
                df= df.replace({col:'unknown'}, mode_col )
                #print(mode_col)
    return df 
    
#Main body
columns=['age','job','marital','education','default','balance','housing',
         'loan','contact','day', 'month', 'duration', 'campaign', 'pdays',
         'previous', 'poutcome', 'label']
data_train=[]    
tree={}
data_test=[] 

#train
train_file= open ( 'train.csv' , 'r' )
for line in train_file: 
    terms= line.strip().split(',')
    data_train.append (terms)
df_train= pd.DataFrame(data_train,columns=columns)

#with unknown
#df_train_m=data_modify(df_train)

#with missing value
df_train_m=data_modify_missing(df_train)

#these should change
max_depth=6
Entropy_method='H'
tree = ID3_func(df_train_m, max_depth, Entropy_method, main_tree=True)
#pprint(tree)

#test
test_file= open ( 'train.csv' , 'r' )
for line in test_file: 
    terms= line.strip().split(',')
    data_test.append (terms)
df_test= pd.DataFrame(data_test,columns=columns)
df_test_M=data_modify(df_test)
percent=test(df_test_M,tree,max_depth)
print(percent)




                    
                
            
        

   
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
        



        
        
