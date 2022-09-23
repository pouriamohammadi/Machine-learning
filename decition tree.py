# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 18:58:06 2021

@author: Pouria
"""
import statistics
import pandas as pd
import numpy as np
from numpy import log
from pprint import pprint
eps = np.finfo(float).eps

def branches(df,columns):
    att_barnches={}
    for att in columns:
        x=df[att].unique()
        att_barnches[att]=x.tolist()
    return(att_barnches)

def entropy_H_GI(fraction, H_M_GI):
    if fraction==0:
        entropy=0
    elif H_M_GI == 'H':
        entropy= -(fraction)*log(fraction)
    elif H_M_GI == 'GI':
        entropy= -(fraction)**2
    return entropy


def IG(df, att, H_M_GI):
    label_vars =df.label.unique()

    #entropy of whole:
    entropy_s=0
    list_M_S=[]
    for labels in label_vars:
        # num= len(df['label'][df.label== labels])
        num=df.loc[df['label']==labels]['weight'].sum()
        num=round(num,6)
        # den = len(df.label)
        den=df.loc[:,'weight'].sum()
        den=round(den,6)
        # print('den', den , 'num', num , labels)
        if num==0 or den==0:
            frac_s=0
        else:
            frac_s=(num/(den))
        #print(num,den)
        if H_M_GI == 'M':
            list_M_S.append(frac_s)
        else:
            entropy_s += entropy_H_GI(frac_s, H_M_GI)
            # print('entropy_s',entropy_s,'\n')
    if H_M_GI == 'GI':
        entropy_s= 1+entropy_s
    if H_M_GI == 'M':
        entropy_s= 1-max(list_M_S)
    # print('entropy_s',entropy_s)    
    # if entropy_s<0 :
        # print(df)

    #entropy of each attribute:
    features=df[att].unique()
    entropy_sigma=0
    #loop over features of each attribute including [job,age,...]
    for feat in features:
        entropy_att=0
        # M
        if H_M_GI == 'M':
            list_M=[]
            #loop over labels of each features to find majority
            for label_var in label_vars:
                # num = len(df[att][df[att]==feat][df.label ==label_var2])
                df_temp=df.loc[df[att]==feat]
                num=df_temp.loc[df_temp.label ==label_var]['weight'].sum()
                num=round(num,6)
                # den = len(df[att][df[att]==feat]) 
                den=df.loc[df[att]==feat]['weight'].sum()
                den=round(den,6)
                if den==0:
                    frac=0
                else:
                    frac = num/(den)
                list_M.append(frac)
                #print(list_M)
            entropy_att = 1-max(list_M)
            
        # H or GI :
        else:
            #loop over labels of each features 
            for label_var in label_vars:
                # num= len(df[att][df[att]==feat][df.label ==label_var])
                df_temp=df.loc[df[att]==feat]
                num=df_temp.loc[df_temp.label ==label_var]['weight'].sum()
                num=round(num,6)
                # den = len(df[att][df[att]==feat]) 
                den=df.loc[df[att]==feat]['weight'].sum()
                den=round(den,4)
                if num==0 or den==0:
                    frac=0
                else:
                    frac = num/(den)
                entropy_att += entropy_H_GI(frac,H_M_GI)
            #Gini Index:
            if H_M_GI == 'GI':
                entropy_att= 1+entropy_att
                
        #sum entropy of a feature
        frac_sigma = den/len(df)
        entropy_sigma += frac_sigma*entropy_att
        # print('entropy of',feat, entropy_att,frac_sigma)
    InformationGain = entropy_s- entropy_sigma
    # print('InformationGain',InformationGain,'\n')
    return InformationGain
        
def choose_node (IG_dict):
    max_IG=0
    #print(IG_dict)
    for attribute in IG_dict:
        if max_IG<=IG_dict[attribute]:
            max_IG=IG_dict[attribute]
            choosen=attribute
        # else:
        #     print(attribute,IG_dict[attribute])
        
    return choosen

def subtable (df,node,branch):
    df_sub=(df[df[node]==branch].reset_index(drop=True))
    return df_sub

def common_func(df):
    x=df.label.unique()
    weight1=df.loc[df['label']==x[0]]['weight'].sum()
    weight2=df.loc[df['label']==x[1]]['weight'].sum()
    if weight1>weight2:
        return x[0]
    else:
        return x[1]
              
def ID3_func(df,depth,H_M_GI,att_barnches,remain_att,main_tree=None,tree=None): 
    
    attributes=[]
    attributes= df.keys()[:-2]
    # print(attributes)
    
    if len(df.label.unique())==1 :
        return df.label[0] 
    
    if depth == 0 and main_tree==None:
        common_label= common_func(df)
        return common_label
    
    #collecting IG of attributes
    else:
        IG_dict={}
        for att in remain_att:
            inf_gain = IG(df, att, H_M_GI)
            IG_dict[att]=inf_gain
        # print(IG_dict)
        
        #finding the node 
        root_node=choose_node(IG_dict)
        # print('root_node', root_node, '\n','remain_att',remain_att, )
        remain_att=remain_att.drop(root_node)
        
        #draw tree
        if tree is None:
            tree={}
            tree[root_node]={}
            
        #check the depth and main_tree  
        if depth == 0 and main_tree==True:
            tree[root_node]=common_func(df)
            return tree
        
        depth=depth-1

        #finding subtable
        node_branches=[]
        node_branches=att_barnches[root_node]
        for branch in node_branches:
            df_sub= subtable (df,root_node,branch)
            if len (df_sub) == 0:
                tree[root_node][branch]=common_func(df)
            else:
                tree[root_node][branch] =ID3_func(df_sub,depth,H_M_GI,att_barnches,remain_att)
    return tree

def predict(tree, df_row,max_depth):
    for key1 in tree.keys():
        # print('keys', tree.keys())
        if type(tree[key1]) is not dict:
            predict_label=tree[key1]
            return predict_label
            # print('xxxxxxx read')
        else:
            tree2=tree[key1]
            for key2 in tree2.keys():
                # print(key2, tree2[key2])
                if type(tree2[key2]) is not dict:
                    # print(df_row[key1], 'key2',key2)
                    if df_row[key1]== key2:
                        predict_label =tree2[key2]
                        # print( 'dfrow',key1, df_row[key1], key2 )
                        # print( 'xxxxxxxxxx predicted')
                        # return predict_label
                        
                else:
                    if df_row[key1]==key2:
                        tree3=tree2[key2]
                        predict_label=predict(tree3, df_row, max_depth)
                        # return predict_label
                        
    # print(predict_label)                
    return predict_label
    
# =============================================================================
# 
#     for key1 in tree.keys():
#         #print (tree[key1])         
#         if type(tree[key1]) is dict:
#             for key2 in tree[key1].keys():
#                 tree2=tree[key1]
#                 if type(tree2[key2]) is not dict:
#                     predict_label= tree2[key2]
#                     if df_row[key1]==key2:
#                         predict_label= tree2[key2]
#                         return predict_label
#             
#                 else:
#                     if df_row[key1]==key2:
#                         tree_new=tree2[key2]   
#                         label_pre= predict(tree_new, df_row,max_depth)
#                         predict_label=label_pre
#                         return predict_label
#                         
#             #make a guess since it is not in the tree           
#             if predict_label is None:
#                 tree_new=tree2[key2]   
#                 label_pre= predict(tree_new, df_row,max_depth)
#                 predict_label=label_pre       
#               
#         else:
#             if max_depth==0:
#                 predict_label= tree[key1]
#                 return predict_label
# 
#             else:
#                 for key2 in tree[key1].keys():
#                     predict_label= key2
#                     if df_row[key1]==tree[key1]:
#                         predict_label= key2
#                         return predict_label
#                     
#     
#     return predict_label
# =============================================================================


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
                # print(col,med)
                for row in range(0,df.shape[0]):
                    if (df.loc[row,col])<=med:
                        df.loc[row,col]=0
                    else:
                        df.loc[row,col]=1
        df=df.replace({'label':'yes'},'+1')
        df=df.replace({'label':'no'},'-1')
        df= df.astype({'label': 'int32'})
        # print (df)
    return df
    
    
#Main body
columns=['age','job','marital','education','default','balance','housing',
         'loan','contact','day', 'month', 'duration', 'campaign', 'pdays',
         'previous', 'poutcome', 'label']
data_train=[]    
tree={}
data_test=[] 

#train dataset
train_file= open ( 'train.csv' , 'r' )
for line in train_file: 
    terms= line.strip().split(',')
    data_train.append (terms)
df_train= pd.DataFrame(data_train,columns=columns)

#test dataset
test_file= open ( 'train.csv' , 'r' )
for line in test_file: 
    terms= line.strip().split(',')
    data_test.append (terms)
df_test= pd.DataFrame(data_test,columns=columns)

#with unknown
df_train_m=data_modify(df_train)

#adding weight to dataset
df_train_m.insert(len(df_train_m.columns),'weight',1)    
# print(df_train_m)

#finding attribute branches
att_barnches=branches(df_train_m,columns)

#modify dataset
df_test_M=data_modify(df_test)

#hyperparameters
max_depth=16

Entropy_method='H'

#ID3
remain_att=[]
remain_att=df_train_m.keys()[:-2] 
tree=ID3_func(df_train_m, max_depth, Entropy_method,att_barnches,remain_att, main_tree=True)
# pprint(tree)

#Accuracy
# percent=test(T,df_test_M,h_list,max_depth,Entropy_method)
percent=test(df_test_M,tree,max_depth)
print('max_depth', max_depth)
print('percent',percent, '\n')



        
