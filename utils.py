# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:22:42 2020

@author: laramos
"""
import re
from missingpy import KNNImputer,MissForest
import numpy as np
import pandas as pd
import os

class Measures:       
    def __init__(self,splits):
        self.clf_auc=np.zeros(splits)
        self.clf_brier=np.zeros(splits)
                        
        self.clf_f1_score=np.zeros(splits)
        self.clf_sens=np.zeros(splits)
        self.clf_spec=np.zeros(splits)
        self.clf_ppv=np.zeros(splits)
        self.clf_npv=np.zeros(splits)
        
        self.clf_fp=np.zeros(splits)
        self.clf_fn=np.zeros(splits)
        self.clf_tp=np.zeros(splits)
        self.clf_tn=np.zeros(splits)
        
        self.clf_r2=np.zeros(splits)
        self.clf_mae=np.zeros(splits) 
        self.clf_mse=np.zeros(splits) 
        self.clf_mdae=np.zeros(splits)
        
        self.clf_r_mae=np.zeros(splits) 
        self.clf_r_mse=np.zeros(splits) 
        self.clf_r_mdae=np.zeros(splits)
        
        self.clf_tpr=list()
        self.clf_fpr=list()
        self.mean_tpr=0.0
        self.run=False
        self.feat_imp=list() 
        self.probas=list()
        self.preds=list()


def Save_fpr_tpr(path_results,names,measures):
    for i in range(0,len(names)): 
        for k in range(0,len(measures[i].clf_fpr)):
            f=np.array(measures[i].clf_fpr[k],dtype='float32')
            t=np.array(measures[i].clf_tpr[k],dtype='float32')
            save_f=path_results+'fpr_'+names[i]+'_'+str(k)
            np.save(save_f,f)
            save_t=path_results+'tpr_'+names[i]+'_'+str(k)
            np.save(save_t,t)  
 
      
def Change_One_Hot(X_train_imp,X_test_imp,vals_mask):
    """
    This function one-hot-encode the features from the vals_mask and returns it as numpy array
    Input:
        frame: original frame with variables
        vals_mask: array of string with the names of the features to be one-hot-encoded [['age','sex']]
    Ouput:
        Result: One-hot-encoded feature set in pd.frame format
    """
    size = X_train_imp.shape[0]
    framen=pd.DataFrame(np.concatenate((X_train_imp,X_test_imp),axis=0),columns=X_train_imp.columns)
    framen_dummies=pd.get_dummies(framen, columns=vals_mask)    
    X_data=np.array(framen_dummies)    
    X_train_imp=(X_data[0:size,:])            
    X_test_imp=(X_data[size:,:])
    cols=framen_dummies.columns
    return(X_train_imp,X_test_imp,cols)


      
def Change_One_Hot_DL(X_train_imp,X_val_imp,X_test_imp,vals_mask):
    """
    This function one-hot-encode the features from the vals_mask and returns it as numpy array
    Input:
        frame: original frame with variables
        vals_mask: array of string with the names of the features to be one-hot-encoded [['age','sex']]
    Ouput:
        Result: One-hot-encoded feature set in pd.frame format
    """
    size_train = X_train_imp.shape[0]
    size_val = X_val_imp.shape[0]
    size_test = X_test_imp.shape[0]
    
    framen=pd.DataFrame(np.concatenate((X_train_imp,
                                        X_val_imp,
                                        X_test_imp),axis=0),columns=X_train_imp.columns)
    
    framen_dummies=pd.get_dummies(framen, columns=vals_mask)    
    X_data=np.array(framen_dummies)    
    X_train_imp=(X_data[0:size_train,:])            
    X_val_imp=(X_data[size_train:(size_train+size_val),:])
    X_test_imp=(X_data[(size_train+size_val):(size_train+size_val+size_test),:])
    cols=framen_dummies.columns

    return(X_train_imp,X_val_imp,X_test_imp,cols)
    
       
    
def Clean_Data(path_data,path_variables):


    frame=pd.read_csv(path_data,sep=';',encoding = "latin",na_values=' ')    
    for var in frame:
        if 'Study' in var:
             frame = frame.rename(columns={var: "StudySubjectID"})
    subj = frame['StudySubjectID']
    Y_mrs = frame['mrs']

    Y_mrs=np.array(Y_mrs,dtype='float32')
    
    Y_tici=frame['posttici_c'].values
    Y_tici=np.array(frame['posttici_c'].factorize(['0','1','2A','2B','2C','3'])[0],dtype="float32")
    #cnonr=frame['cnonr']
    
    miss_mrs=Y_mrs<0
    Y_mrs[miss_mrs]=np.nan
    miss_tici=Y_tici<0
    Y_tici[miss_tici]=np.nan
    
    var=pd.read_csv(path_variables)
    
    mask_cont = var[var['type']=='cont']['names']
    mask_cat = var[var['type']=='cat']['names']
    
    var=var.dropna(axis=0)

    frame=frame[var['names']]   
   
    #These are categorical with multiple categories
    vals_mask=['occlsegment_c_cbs','cbs_occlsegment_recoded','ct_bl_leukd']  
    
    cols=frame.columns
    
    data=np.zeros((frame.shape))
    
    #this features have commas instead of points for number, ruins the conversion to float
    frame['glucose']=frame['glucose'].apply(lambda x: str(x).replace(',','.'))
    frame['INR']=frame['INR'].apply(lambda x: str(x).replace(',','.'))
    frame['crpstring']=frame['crpstring'].apply(lambda x: str(x).replace(',','.'))
    frame['age']=frame['age'].apply(lambda x: str(x).replace(',','.'))
    
    #smoking =2 is missing/   prev_str =2 is missing    ivtrom =2 is missing               

    #frame['ivtrom']=frame['ivtrom'].replace(9,np.nan)
    frame['ivtci']=frame['ivtci'].replace(9,np.nan)
    frame['inhosp']=frame['inhosp'].replace(9,np.nan)
    frame['smoking']=frame['smoking'].replace(2,np.nan)
    frame['prev_str']=frame['prev_str'].replace(2,np.nan)
    frame['NIHSS_BL']=frame['NIHSS_BL'].replace(-1,np.nan)
    frame['ASPECTS_BL']=frame['ASPECTS_BL'].replace(-1,np.nan)

    for i in range(0,frame.shape[1]):
        #if frame.cols[i].dtype.name=='category':
        if var.iloc[i]['type']=='cat':
            frame[cols[i]]=frame[cols[i]].astype('category')           
            cat=frame[cols[i]].cat.categories
            frame[cols[i]],l=frame[cols[i]].factorize([np.nan,cat])
            data[:,i]=np.array(frame[cols[i]],dtype="float32")
            data[data[:,i]==-1,i]=np.nan  
        else:
            data[:,i]=np.array(frame[cols[i]],dtype="float32")
            data[data[:,i]==-1,i]=np.nan     
    

    miss=np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        miss[i]=np.count_nonzero(np.isnan(data[:,i]))
    
    #return(frame,cols,var,data,Y_mrs,Y_tici,data_img)
    return(frame,cols,var,data,Y_mrs,Y_tici,vals_mask,miss,subj,mask_cont,mask_cat)
    
    
                        
    
def Impute_Data_MICE(X_train,y_train,X_test,y_test,n_imputations,vals_mask,cols,mrs, i):
    
    XY_incomplete = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)       
    XY_completed_train = []
    XY_completed_test = []
    
    for i in range(n_imputations):
        imputer = IterativeImputer(sample_posterior=True, random_state=i*10,initial_strategy="mean",min_value=0)
        XY_completed_train.append(imputer.fit_transform(XY_incomplete))
        XY_completed_test.append(imputer.transform(np.concatenate((X_test,y_test.reshape(-1,1)),axis=1)))
        
        if mrs:            
            X_train_imp=(XY_completed_train[i][:,0:data.shape[1]])
            y_train_imp=np.array(XY_completed_train[i][:,data.shape[1]]<=2,dtype="int16")
            X_test_imp=(XY_completed_test[i][:,0:data.shape[1]])
            y_test_imp=np.array(XY_completed_test[i][:,data.shape[1]]<=2,dtype="int16")
        else:
            X_train_imp=(XY_completed_train[i][:,0:data.shape[1]])
            y_train_imp=np.array(XY_completed_train[i][:,data.shape[1]]<3,dtype="int16")
            X_test_imp=(XY_completed_test[i][:,0:data.shape[1]])
            y_test_imp=np.array(XY_completed_test[i][:,data.shape[1]]<3,dtype="int16")
        
        for j in range(0,X_train_imp.shape[1]):
            if  var.iloc[j]['type']=='cat':
                X_train_imp[:,j]=np.clip(np.round(X_train_imp[:,j]),min_vals[j],max_vals[j])
                X_test_imp[:,j]=np.clip(np.round(X_test_imp[:,j]),min_vals[j],max_vals[j])
            else:
                X_train_imp[:,j]=np.round(X_train_imp[:,j],decimals=1)
                X_test_imp[:,j]=np.round(X_test_imp[:,j],decimals=1)
                
                    
        return(X_train_imp,y_train_imp,X_test_imp,y_test_imp)   
        
def Impute_Data(X_train,y_train,X_test,y_test,n_neighbors,imputer,cat_vars,min_vals,max_vals,var):
    
    origin_shape = X_train.shape
    
    XY_incomplete_train = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)       
    XY_incomplete_test = np.concatenate((X_test,y_test.reshape(-1,1)),axis=1)
    
    if imputer=='KNN':
        imputer = KNNImputer(n_neighbors=n_neighbors)
    else:
        imputer = MissForest(random_state=1,n_jobs=-1)
        
    XY_completed_train = imputer.fit_transform(XY_incomplete_train,cat_vars = np.array(cat_vars))
    XY_completed_test = imputer.transform(XY_incomplete_test)
          
    X_train_imp=(XY_completed_train[:,0:origin_shape[1]])
    y_train_imp_orig=np.array(XY_completed_train[:,origin_shape[1]],dtype="int16")
    y_train_imp=np.array(XY_completed_train[:,origin_shape[1]],dtype="int16")
    X_test_imp=(XY_completed_test[:,0:origin_shape[1]])
    y_test_imp=np.array(XY_completed_test[:,origin_shape[1]],dtype="int16")
    y_test_imp_orig=np.array(XY_completed_test[:,origin_shape[1]],dtype="int16")

        
    for j in range(0,X_train_imp.shape[1]):
        if  var.iloc[j]['type']=='cat':
            X_train_imp[:,j]=np.clip(np.round(X_train_imp[:,j]),min_vals[j],max_vals[j])
            X_test_imp[:,j]=np.clip(np.round(X_test_imp[:,j]),min_vals[j],max_vals[j])
        else:
            X_train_imp[:,j]=np.round(X_train_imp[:,j],decimals=0)
            X_test_imp[:,j]=np.round(X_test_imp[:,j],decimals=0)
    
    #min_vals_imp=np.nanmin(np.concatenate((X_train_imp,X_test_imp),axis=0),axis=0)
    #max_vals_imp=np.nanmax(np.concatenate((X_train_imp,X_test_imp),axis=0),axis=0)  
                    
    return(X_train_imp,y_train_imp,X_test_imp,y_test_imp,y_train_imp_orig,y_test_imp_orig)  
    
def Impute_Data_DL(X_train,X_test,X_val,n_neighbors,imputer,mask_cont,min_vals,max_vals,var):
    
    X_train = frame_clin_train
    X_val = frame_clin_val
    X_test = frame_clin_test
    #cat_vars = np.array(cont_vars_pos).reshape(-1,1)
    
    origin_shape = X_train.shape

    subj_train = X_train['ID']
    subj_val = X_val['ID']
    subj_test = X_test['ID']
    
    X_train = X_train.drop('ID',axis=1)
    X_test = X_test.drop('ID',axis=1)
    X_val = X_val.drop('ID',axis=1)
    
    orig_cols = X_train.columns
    
    XY_incomplete_train =np.array(X_train)
    XY_incomplete_val =np.array(X_val)
    XY_incomplete_test = np.array(X_test)
    
    cont_vars_pos, cat_vars_pos = get_pos_cont_and_cat_variables(mask_cont,X_train.columns)
    min_vals = np.nanmin(X_train,axis=0)
    max_vals = np.nanmax(X_train,axis=0)

    
    if imputer=='KNN':
        imputer = KNNImputer(n_neighbors=n_neighbors)
    else:
        imputer = MissForest(random_state=1,n_jobs=-1)
        
    XY_completed_train = imputer.fit_transform(XY_incomplete_train,cat_vars = np.array(cat_vars_pos))
    XY_completed_test = imputer.transform(XY_incomplete_test)
    XY_completed_val = imputer.transform(XY_incomplete_val)
          
    X_train_imp=(XY_completed_train[:,0:origin_shape[1]])
    X_test_imp=(XY_completed_test[:,0:origin_shape[1]])
    X_val_imp=(XY_completed_val[:,0:origin_shape[1]])

        
    for j in range(0,X_train_imp.shape[1]):
        #if  var.iloc[j]['type']=='cat':
        if  j in cat_vars_pos:
            X_train_imp[:,j]=np.clip(np.round(X_train_imp[:,j]),min_vals[j],max_vals[j])
            X_test_imp[:,j]=np.clip(np.round(X_test_imp[:,j]),min_vals[j],max_vals[j])
            X_val_imp[:,j]=np.clip(np.round(X_val_imp[:,j]),min_vals[j],max_vals[j])
        else:
            X_train_imp[:,j]=np.round(X_train_imp[:,j],decimals=0)
            X_test_imp[:,j]=np.round(X_test_imp[:,j],decimals=0)
            X_val_imp[:,j]=np.round(X_val_imp[:,j],decimals=0)
    
    X_train_imp = pd.DataFrame(X_train_imp,columns = orig_cols)
    X_val_imp = pd.DataFrame(X_val_imp,columns = orig_cols)
    X_test_imp = pd.DataFrame(X_test_imp,columns = orig_cols)
    X_train_imp['ID'] = subj_train
    X_val_imp['ID'] = subj_val
    X_test_imp['ID'] = subj_test
                    
    return(X_train_imp,X_val_imp,X_test_imp)  
          
def get_pos_cont_and_cat_variables(mask_cont,cols_o):
    cont_vars_pos = list()
    mask_cont = list(mask_cont)
    for i in range(0,len(mask_cont)):
            cont_vars_pos.append(np.where(mask_cont[i]==cols_o)[0][0])        
    cat_vars_pos = (list(set(np.arange(0,len(cols_o))) - set(cont_vars_pos)))
    return(cont_vars_pos,cat_vars_pos)


def get_ids_images(path):
    id_done_list=list()
    done = sorted(os.listdir(path))
    for f in done:
        id_p = re.search('R[0-9][0-9][0-9][0-9]',f)
        if id_p:
            id_done_list.append(id_p[0])  
    return(id_done_list)


def clean_mask(vals_mask,cols_o):
    vals = list()
    for v in vals_mask:
        if v in cols_o:
            vals.append(v)
    vals_mask = vals  
    return(vals_mask)
