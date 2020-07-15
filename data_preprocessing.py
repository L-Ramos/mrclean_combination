# -*- coding: utf-8 -*-
"""
This file contains fucntions for pre-processin the MR clean dataset. The functions are specific for the dataset we used. Mr Clean part 1

@author: laramos
"""

import numpy as np
import pandas as pd

def Change_One_Hot(X_train_imp,X_test_imp,vals_mask,cols):
    """
    This function one-hot-encode the features from the vals_mask and returns it as numpy array
    Input:
        frame: original frame with variables
        vals_mask: array of string with the names of the features to be one-hot-encoded [['age','sex']]
    Ouput:
        Result: One-hot-encoded feature set in pd.frame format
    """
    size = X_train_imp.shape[0]
    framen=pd.DataFrame(np.concatenate((X_train_imp,X_test_imp),axis=0),columns=cols)
    framen_dummies=pd.get_dummies(framen, columns=vals_mask)    
    X_data=np.array(framen_dummies)    
    X_train_imp=(X_data[0:size,:])            
    X_test_imp=(X_data[size:,:])
    cols=framen_dummies.columns
    return(X_train_imp,X_test_imp,cols)
# =============================================================================
#     bool_mask=np.zeros(cols.shape[0],dtype="bool")
#     
#     for k in range(len(vals_mask)):
#         for i in range(cols.shape[0]):
#             if cols[i]==vals_mask[k]:               
#                 bool_mask[i]=True
#                 
#     X_vars=np.array(X_train_imp,dtype='float64')
#     X_vars_test=np.array(X_test_imp,dtype='float64')
#     rf_enc = OneHotEncoder(categorical_features=bool_mask)
#     rf_enc.fit(X_vars)
#     Result=rf_enc.transform(X_vars)
#     Result_test=rf_enc.transform(X_vars_test)
#     Result=Result.toarray()
#     Result=np.array(Result,dtype='float64')
#     Result_test=Result_test.toarray()
#     Result_test=np.array(Result_test,dtype='float64')
    #return(Result,Result_test)
# =============================================================================
            
    

def Clean_Data(path_data,path_variables):
    
    #frame=pd.io.stata.read_stata(path_data,encoding='ISO-8859-1')      
    #frame=pd.read_stata(path_data,encoding ="ISO-8859-1") 
    #frame = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\MrClean_Poor\data\data_complete.csv",sep=';',encoding = "latin",na_values=' ')
    frame=pd.read_csv(path_data,sep=';',encoding = "latin",na_values=' ')
    
    subj = frame['StudySubjectID']
    #result=list()
    #for i in range(len(cols)):
    #    if 'CBS'in cols[i] or 'cbs' in cols[i]:
    #        result.append(cols[i])
    
    Y_mrs = frame['mrs']
    original_mrs = Y_mrs
    
    #frame = frame.fillna(np.nan)
    
    #center=np.array(frame['Centrumnummer'].factorize()[0],dtype="float32")

    Y_mrs=np.array(Y_mrs,dtype='float32')
    
    Y_tici=frame['posttici_c'].values
    Y_tici=np.array(frame['posttici_c'].factorize(['0','1','2A','2B','2C','3'])[0],dtype="float32")
    #cnonr=frame['cnonr']
    
    miss_mrs=Y_mrs<0
    Y_mrs[miss_mrs]=np.nan
    miss_tici=Y_tici<0
    Y_tici[miss_tici]=np.nan
    
    var=pd.read_csv(path_variables)
    var=var.dropna(axis=0)
     
    #for i in range(0,len(var)):
    #    var.iloc[i]['names']=str(var.iloc[i]['names']).replace(" ","")
          
    frame=frame[var['names']]   

    """
    for i in range(frame.shape[1]):
        for j in range(frame.shape[0]):
            if type(frame.iloc[j][i])==str:
                if ',' in frame.iloc[j][i]:
                    print(i)
    """            
    for i in range(0,frame.shape[0]):
        if frame.iloc[i]['ct_bl_leuk']==0:
            frame.set_value(i,'ct_bl_leukd',0)
            
    for i in range(0,frame.shape[0]):
        if frame.iloc[i]['ivtrom']==1:            
            frame.set_value(i,'ivtci',0)
    frame.drop(['ivtrom','ct_bl_leuk'], axis=1)
    
    vals_mask=['premrs','collaterals','occlsegment_c','cbs_occlsegment_recoded','occlside_c', 'ct_bl_leukd']  # nihssbl_afa, nihssbl_gaze
    
    #vals_mask_complete=['premrs','ASPECTS_BL','occlsegment_c','collaterals','cbs_occlsegment_recoded','CBS_BL','NIHSS_BL','gcs','ct_bl_leukd']
    
    
    cols=frame.columns
    
    data=np.zeros((frame.shape))
    
    #this features have commas instead of points for number, ruins the conversion to float
    frame['glucose']=frame['glucose'].apply(lambda x: str(x).replace(',','.'))
    frame['INR']=frame['INR'].apply(lambda x: str(x).replace(',','.'))
    frame['crp']=frame['crp'].apply(lambda x: str(x).replace(',','.'))
    
    #smoking =2 is missing/   prev_str =2 is missing    ivtrom =2 is missing               

    frame['ivtrom']=frame['ivtrom'].replace(9,np.nan)
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
        
    
    mask_img=np.array(var['img'],dtype='bool')
    data_img=data[:,mask_img]
    #return(frame,cols,var,data,Y_mrs,Y_tici,data_img)
    return(frame,cols,var,data,Y_mrs,Y_tici,data_img,vals_mask,miss,original_mrs,subj)
    
#X_train_imp = X_train_imp[:,t]
#X_test_imp = X_test_imp[:,t]
#X_train_imp = X_train_imp[:,[29,36]]
#X_test_imp = X_test_imp[:,[29,36]]
#if l==0:
#   histoffeats,meanauc,stdauc = Select_Recursive_Features  (X_train_imp,y_train,5)
#else:
#   h,meanauc,stdauc = Select_Recursive_Features  (X_train_imp,y_train,5)
#   histoffeats=histoffeats+h
#   t = histoffeats>=75
    
   
#scaler = StandardScaler().fit(X_train_imp)
#scaler = MinMaxScaler().fit(X_train_imp)
#scaler = Normalizer().fit(X_train_imp)
#X_train_imp=scaler.transform(X_train_imp)
#X_test_imp=scaler.transform(X_test_imp)

#print("Train Ratio",np.sum(y_train)/y_train.shape[0])
#print("Test Ratio",np.sum(y_test)/y_test.shape[0])

#from sklearn import svm
#from  sklearn.metrics import mean_absolute_error
#from sklearn.linear_model import LinearRegression
#clf = svm.SVR()                    
#clf.fit(X_train_imp, y_train_orig) 
#preds = clf.predict(X_test_imp)
#error_svm = mean_absolute_error(y_test_orig,preds)
#
#clf = LinearRegression()                    
#clf.fit(X_train_imp, y_train_orig) 
#preds_lr = clf.predict(X_test_imp)
#error_lr = mean_absolute_error(y_test_orig,preds_lr)

  