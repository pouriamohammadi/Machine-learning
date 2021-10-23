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
import matplotlib.pyplot as plt

# eps = np.finfo(float).eps

def entropy_H_GI(fraction, H_M_GI):
    if fraction==0:
        entropy=0
    elif H_M_GI == 'H':
        entropy= -(fraction)*log(fraction)
    elif H_M_GI == 'GI':
        entropy= -(fraction)**2
    return entropy

def branches(df,columns):
    att_barnches={}
    for att in columns:
        x=df[att].unique()
        att_barnches[att]=x.tolist()
    return(att_barnches)

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
                den=round(den,6)
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
    
def ID3_func(df,depth,H_M_GI,att_barnches,remain_att,feature_subset,main_tree=None,tree=None): 
    # attributes=[]
    # attributes= df.keys()[:-2]
    
    if len(df.label.unique())==1 and main_tree==None:
        return df.label[0] 
    
    if depth == 0 and main_tree==None:
        common_label= common_func(df)
        return common_label
    #collecting IG of attributes
    else:
        IG_dict={}
        # print(remain_att)
        if len(remain_att)>feature_subset:
            # random_att=remain_att.sample(5,replace=False,axis=0)
            random_att = np.random.choice(remain_att, size=feature_subset, replace=False)
            # print(random_att)
        else:
            random_att=remain_att
        for att in random_att:
            inf_gain = IG(df, att, H_M_GI)
            IG_dict[att]=inf_gain
        #print(IG_dict)
        
        #finding the node 
        root_node=choose_node(IG_dict)
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
                tree[root_node][branch] =ID3_func(df_sub,depth,H_M_GI,att_barnches,remain_att,feature_subset)
    return tree

def predict(tree, df_row,max_depth):
    for key1 in tree.keys():
        if type(tree[key1]) is not dict:
            predict_label=tree[key1]
            return predict_label
        else:
            tree2=tree[key1]
            for key2 in tree2.keys():
                if type(tree2[key2]) is not dict:
                    if df_row[key1]== key2:
                        predict_label =tree2[key2]
                else:
                    if df_row[key1]==key2:
                        tree3=tree2[key2]
                        predict_label=predict(tree3, df_row, max_depth)
    return predict_label

def data_modify(df):
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

def bagging_train(T, df, max_depth, Entropy_method,att_barnches):
    h_list=[]
    for t in range (1,T+1):
        remain_att=[]
        remain_att=df.keys()[:-2] 
        df_new=df.sample( 1000 ,replace=False,axis=0)
        df_new=df_new.reset_index(drop=True)
        tree=ID3_func(df_new, max_depth, Entropy_method,att_barnches,remain_att, main_tree=True)
        h_list.append(tree)
    return h_list

def randomforrest(T, df_train, max_depth, Entropy_method,att_barnches,feature_subset):
    h_list=[]
    for t in range(1,T+1):
        remain_att=[]
        remain_att=df_train.keys()[:-2] 
        #sampling
        df_new=df_train.sample(1000 ,replace=False,axis=0)
        df_new=df_new.reset_index(drop=True)
        #finding tree
        tree=ID3_func(df_new, max_depth, Entropy_method,att_barnches,remain_att,feature_subset,main_tree=True)
        h_list.append(tree)
    return h_list
    
def bias_variance(n,T,df,whole_bagged_trees,single_trees,max_depth):
    #list for bagged trees
    list_bias_bag=[]
    lsit_var_bag=[]
    #list for single trees
    list_bias=[]
    lsit_var=[]
    for row in range (0,df.shape[0]):
        df_rows = df.iloc[row , : ]
        pr_label_list_bag=[]
        pr_label_list=[]
        for i in range(0, n):
            #predict for single trees
            single_tree=single_trees[i]
            pr_label= predict(single_tree,df_rows,max_depth)
            pr_label_list.append(pr_label)
            #predict for bagged trees
            h_list=whole_bagged_trees[i]
            H=0
            for j in range (0,T):
                tree = h_list[j]
                h=predict(tree, df_rows,max_depth)
                # print('h', h)
                H= H + h
            pr_label_bag=int(np.sign(H))
            if pr_label_bag==0:
                pr_label_bag=+1
            pr_label_list_bag.append(pr_label_bag)
            
        # print('pr_label_list',pr_label_list, 'pr_label_list_bag', pr_label_list_bag)    
        #E single trees
        E_predict= np.mean(pr_label_list)
        #E bagged trees
        E_predict_bag= np.mean(pr_label_list_bag)
        # print('E_predict_bag',E_predict_bag, 'E_predict',E_predict)
        real_label= df_rows.label
        
        #calculate bias & variance single trees
        bias= (real_label-E_predict)**2
        var=np.var(pr_label_list)*(len(pr_label_list))/(len(pr_label_list)-1)
        list_bias.append(bias)
        lsit_var.append(var)
        #calculate bias & variance bagged trees
        bias_bag= (real_label-E_predict_bag)**2
        var_bag=np.var(pr_label_list_bag)*(len(pr_label_list_bag))/(len(pr_label_list_bag)-1)
        list_bias_bag.append(bias_bag)
        lsit_var_bag.append(var_bag)
        # print ('bias',bias)
        # print('pr_label_list',pr_label_list)
    
    #expected error single  
    bias_mean= statistics.mean(list_bias)
    variance_mean= statistics.mean(lsit_var)
    expected_error= bias_mean+variance_mean
    #expected error bagged  
    bias_mean_bag= np.mean(list_bias_bag)
    variance_mean_bag= np.mean(lsit_var_bag)
    expected_error_bag= bias_mean_bag+variance_mean_bag
    print ('Single trees : bias_mean= {} ,variance_mean= {}'.format(bias_mean,variance_mean))
    print ('Forrest trees : bias_mean_bag= {} , variance_mean_bag= {}'.format(bias_mean_bag,variance_mean_bag))
    return expected_error_bag , expected_error

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

#test
test_file= open ( 'test.csv' , 'r' )
for line in test_file: 
    terms= line.strip().split(',')
    data_test.append (terms)
df_test= pd.DataFrame(data_test,columns=columns)

#modify with unknown
df_train_m=data_modify(df_train)
df_test_M=data_modify(df_test)

#adding weight to dataset
df_train_m.insert(len(df_train_m.columns),'weight',1)    
# print(df_train_m)

#finding attribute branches
att_barnches=branches(df_train_m,columns)

#hyperparameters
max_depth=16
Entropy_method='H'
n = 100
T= 100
feature_subset=2

single_trees=[]
whole_bagged_trees=[]
for i in range (0,n):
    h_list = randomforrest(T, df_train_m, max_depth, Entropy_method,att_barnches,feature_subset)
    single_trees.append(h_list[0])
    whole_bagged_trees.append(h_list)
    
#bias and variance
E_error_bagged ,E_error_tree = bias_variance(n,T,df_test_M,whole_bagged_trees,single_trees,max_depth)
print('E_error_tree ', E_error_tree , 'E_error_bagged',E_error_bagged,'\n')
   
# plt.title('error')
# plt.plot(T_list,error_list)
# plt.show()   


        
