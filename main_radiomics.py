
"""
Created on Mon Dec 10 15:57:11 2018

#TODO:          
        CREATE ANOTHER FILE FOR THE FEATURE EnGINEERING 
        run some Pca to check results
        try folds so I can report confusion matrix
        aspects, nihss, maybe devide into groups? or non linear relationship
        Add LASSO and XGB
        Create NN in tensorflow and optimization pipeline
        
        
        

@author: laramos
"""

import warnings
warnings.filterwarnings("ignore")
#from fancyimpute import IterativeImputer, KNN
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
#import seaborn as sns
import pickle

import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
import time
from sklearn.metrics import roc_auc_score
import Methods as mt
#import Methods_regression as mt_r
import utils as ut

import xgboost as xgb
import shap


cwd = os.getcwd()
os.chdir(cwd)



def rename_image_features(frame_img,image_columns):
    for i,val in enumerate(image_columns):
        if val!='ID':
            image_columns[i]='feat'+str(val)
    frame_img.columns = image_columns
    #image_columns.pop()
    return(frame_img)

def remove_failed_path(frame,folder_path,id_col):
    #removes the features from images that have a folder but failed (failed was defined by visual inspection)
    id_done_part1 = ut.get_ids_images(r'L:\basic\Personal Archive\L\laramos\Disk E\MrClean_part1')
    id_done_part2 = ut.get_ids_images(r'L:\basic\Personal Archive\L\laramos\Disk E\MrClean_part2')
    comb_ids = id_done_part1 +id_done_part2
    np.save(os.path.join(folder_path,'all_clin_ids'),comb_ids)
    
    f_part1 = pd.read_csv(r'\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\imaging/failed_part1.csv')
    f_part2 = pd.read_csv(r'\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\imaging/failed_part2.csv')
    f_part2 = f_part2[f_part2.status=='Failed']
    failed_ids = list(f_part1['s_ids'])+list(f_part2['s_ids'])
    np.save(os.path.join(folder_path,'failed_img_ids'),failed_ids)
            
    all_ids = [i for i in comb_ids if i not in failed_ids]    
    #frame_failed = frame[~frame.s_id.isin(all_ids)]
    frame = frame[frame[id_col].isin(all_ids)]
    return(frame)
 
def remove_failed_file(frame,folder_path,id_col):
    #removes the features from images that have a folder but failed (failed was defined by visual inspection)
 
    comb_ids = np.load(os.path.join(folder_path,'all_clin_ids.npy'))
    
    failed_ids = np.load(os.path.join(folder_path,'failed_img_ids.npy'))
            
    all_ids = [i for i in comb_ids if i not in failed_ids]    
    #frame_failed = frame[~frame.s_id.isin(all_ids)]
    frame = frame[frame[id_col].isin(all_ids)]
    return(frame)   

    
def remove_correlated(frame,id_col,plot=False, threshold=0.99,clin=False):
    # remove columns that are too correlate
    subj = frame[id_col]
    frame = frame.drop(id_col,axis=1)
    if clin:
        Y_mrs = np.array(frame.y_mrs)
        frame = frame.drop('y_mrs',axis=1)
    corr = frame.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
   
    # Draw the heatmap with the mask and correct aspect ratio
    if plot:
        f, ax = plt.subplots(figsize=(25, 15))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    corr_matrix = frame.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] >threshold) or sum(frame[column])==0]
    
    frame=frame.drop(frame[to_drop], axis=1)
    #if clin:
    #    frame['y_mrs'] = Y_mrs
    frame[id_col] = subj
    return(frame)
    


             
def save_imputed_data(path_imp_data,X_train_imp,X_test_imp,y_train,y_test,l,subj_train,subj_test):
    X_train_imp = pd.DataFrame(X_train_imp,columns=cols_o[0:X_train_imp.shape[1]])
    X_test_imp = pd.DataFrame(X_test_imp,columns=cols_o[0:X_test_imp.shape[1]])
    
    X_train_imp['s_id'] = subj_train.values
    X_train_imp['y_mrs'] = y_train
    X_test_imp['s_id'] = subj_test.values
    X_test_imp['y_mrs'] = y_test
    
    X_train_imp.to_csv(os.path.join(path_imp_data,('imp_data_train'+str(l)+".csv")),index=False)
    X_test_imp.to_csv(os.path.join(path_imp_data,('imp_data_test'+str(l)+".csv")),index=False)
    np.save(os.path.join(path_imp_data,('imp_y_train'+str(l)+".npy")),y_train)
    np.save(os.path.join(path_imp_data,('imp_y_test'+str(l)+".npy")),y_test) 
    
def load_imputed_data(path_imp_data,l):   
    X_train_imp = pd.read_csv(path_imp_data+'imp_data_train'+str(l)+".csv")
    X_test_imp = pd.read_csv(path_imp_data+'imp_data_test'+str(l)+".csv")
    subj_train = X_train_imp['s_id']
    subj_test = X_test_imp['s_id']
    y_train = np.load(path_imp_data+'imp_y_train'+str(l)+".npy")
    y_test = np.load(path_imp_data+'imp_y_test'+str(l)+".npy")
    X_train_imp = X_train_imp.drop(X_train_imp[['s_id','y_mrs']], axis=1)
    X_test_imp = X_test_imp.drop(X_test_imp[['s_id','y_mrs']], axis=1)
    
    return(X_train_imp,X_test_imp,y_train,y_test,subj_train,subj_test)
                    
def merge_clinical_image_data(subj_train,subj_test,X_train_imp,X_test_imp,y_train,y_test,frame_img,cols):
    
    X_train_imp = pd.DataFrame(X_train_imp,columns=clin_vars)
    X_test_imp = pd.DataFrame(X_test_imp,columns=clin_vars)
    
    X_train_imp['s_id'] = subj_train.values
    X_train_imp['y_mrs'] = y_train
    X_test_imp['s_id'] = subj_test.values
    X_test_imp['y_mrs'] = y_test
    
    X_train_imp = X_train_imp.merge(frame_img,left_on='s_id',right_on='ID')
    X_test_imp = X_test_imp.merge(frame_img,left_on='s_id',right_on='ID')
    
    y_train = np.array(X_train_imp['y_mrs'])
    y_test = np.array(X_test_imp['y_mrs'])
    X_train_imp = X_train_imp.drop(['s_id','y_mrs','ID'], axis=1)
    X_test_imp = X_test_imp.drop(['s_id','y_mrs','ID'], axis=1)
    
    return(X_train_imp,X_test_imp,y_train,y_test)
                    
def fix_create_paths(name,imp,opt,und,data_to_use,frame_img):
    path_results=(r".//"+name+imp+"_opt-"+opt+"_und-"+und+"//")
    path_results_main = path_results
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    if 'clinical' in data_to_use:
        frame_img = frame_img['ID']
        path_results = os.path.join(path_results,data_to_use)
        if not os.path.exists(path_results):            
            os.makedirs(path_results)
    else:
        if data_to_use=='both':
            path_results = path_results+r'\combination\\'
        else:
            if data_to_use=='image':
                path_results = path_results+r'\image\\'
        if not os.path.exists(path_results):
            os.makedirs(path_results) 
            
            
    path_imp_data = path_results_main+r'data//'
    if not os.path.exists(path_imp_data):
        os.makedirs(path_imp_data)
        need_imp = True
    else:
        need_imp = False
    return(path_results,path_results_main,path_imp_data,need_imp,frame_img)


#folder_path = r"/home/ubuntu"
folder_path = r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\data"

path_data = os.path.join(folder_path,"complete_data_part2.csv")

path_variables = os.path.join(folder_path,"Baseline_contscore_new_review.csv")

_,cols_o,var,data,Y_mrs,Y_tici,vals_mask,miss,subj,mask_cont,mask_cat=ut.Clean_Data(path_data,path_variables)

cont_vars_pos, cat_vars_pos = ut.get_pos_cont_and_cat_variables(mask_cont,cols_o)

frame_clin = pd.DataFrame(data,columns=cols_o)
frame_clin['s_id'] = subj
frame_clin['y_mrs'] = Y_mrs

frame_clin = remove_correlated(frame_clin,id_col='s_id',plot=False,threshold=0.85,clin=True)
frame_clin = frame_clin.drop(['s_id'], axis=1)

vals_mask = ut.clean_mask(vals_mask,cols_o)
mask_cont = ut.clean_mask(mask_cont,cols_o)
mask_cat = ut.clean_mask(mask_cat,cols_o)

clin_vars = list(mask_cont) + list(mask_cat)

data=np.array(frame_clin)
cols_o = frame_clin.columns
min_vals = np.nanmin(data,axis=0)
max_vals = np.nanmax(data,axis=0)

     
print("Clinical Data Loaded and Cleaned")

frame_img = pd.read_csv(os.path.join(folder_path,"image_features_complete.csv"))

frame_img = rename_image_features(frame_img,list(frame_img.columns))

frame_img = remove_correlated(frame_img,id_col='ID',plot=False,threshold=0.85,clin=False)

frame_img = remove_failed_file(frame_img,folder_path,id_col='ID')

mask_cont = list(mask_cont)
cols_img = list(frame_img.columns)
cols_img.pop()
mask_cont_all = mask_cont + cols_img
print("Image Data Loaded and Cleaned")

#clean the clinical data now so we can use proper train and test splits
frame_clin['s_id'] = subj
frame_clin['y_mrs'] = Y_mrs
f = frame_img['ID']
frame_clin = frame_clin.merge(f,left_on='s_id',right_on='ID')
frame_clin = frame_clin.drop_duplicates()
subj = frame_clin['s_id']
Y_mrs = np.array(frame_clin['y_mrs'])
frame_clin = frame_clin.drop(['s_id','y_mrs','ID'], axis=1)

data = np.array(frame_clin)
cols_o = frame_clin.columns
min_vals = np.nanmin(data,axis=0)
max_vals = np.nanmax(data,axis=0)
 
und='W'
opt='roc_auc'

splits=10
cv=5

mean_tprr = 0.0

rfc_m = ut.Measures(splits)
svm_m = ut.Measures(splits)
lr_m = ut.Measures(splits)
xgb_m = ut.Measures(splits)
nn_m = ut.Measures(splits)       

#for imp in imputation:
imp='RF'
#for knn imputation
n_neighbors=5
#data_options = ['clinical','both','image']
data_options = ['both','image']
#data_options = ['clinical_no_scores','clin_no_scores_with_autofeats']\
#data_options = ['clin_no_scores_with_autofeats']    
    
#regression or classification
task = 'classification'
#task = 'regression'


for data_to_use in data_options:
    
    path_results,path_results_main,path_imp_data,need_imp,frame_img = fix_create_paths(task,imp,opt,und,data_to_use,frame_img)
                     
    start_pipeline = time.time()
                
    sk = KFold(n_splits=splits, shuffle=True, random_state=1)                
    
    for l, (train_index,test_index) in enumerate(sk.split(data, Y_mrs)):
        
        if need_imp:
            X_train, X_test = data[train_index,:], data[test_index,:]
            y_train, y_test = Y_mrs[train_index], Y_mrs[test_index] 
            subj_train,subj_test = subj[train_index], subj[test_index]
            
            print("Imputing data! Iteration = ",l) 
                
            if imp=='MICE':
                X_train_imp,y_train,X_test_imp,y_test=ut.Impute_Data_MICE(X_train,y_train,X_test,y_test,1,vals_mask,cols_o,True,l) 
            else:                                                                            
                X_train_imp,y_train,X_test_imp,y_test,y_train_orig,y_test_orig=ut.Impute_Data(X_train,y_train,X_test,y_test,n_neighbors,imp,cat_vars_pos,min_vals,max_vals,var)
                                       
                save_imputed_data(path_imp_data,X_train_imp,X_test_imp,y_train,y_test,l,subj_train,subj_test)                                  
        else:
            print('Found imputation files! Loading.')
            
        X_train_imp,X_test_imp,y_train,y_test,subj_train,subj_test = load_imputed_data(path_imp_data,l)

        X_train_imp, X_test_imp,y_train,y_test = merge_clinical_image_data(subj_train,subj_test,X_train_imp,X_test_imp,y_train,y_test,frame_img,clin_vars) 
        
        X_train_imp,X_test_imp,cols_recoded = ut.Change_One_Hot(X_train_imp,X_test_imp,vals_mask)
         
        if 'clinical' in data_to_use:
            scaler = ColumnTransformer([('norm1', StandardScaler(),mask_cont)], remainder='passthrough')
        else:
            scaler = ColumnTransformer([('norm1', StandardScaler(),mask_cont_all)], remainder='passthrough')
                            
        f_train = pd.DataFrame(X_train_imp,columns=cols_recoded)
        f_test = pd.DataFrame(X_test_imp,columns=cols_recoded)
         
        scaler = scaler.fit(f_train)    
        X_train_imp = scaler.transform(f_train)
        X_test_imp = scaler.transform(f_test)
        
        mask_cat_recoded = [f for f in cols_recoded if f not in mask_cont]
        final_cols = mask_cont.copy()
        final_cols.extend(mask_cat_recoded)
    
        f_train = pd.DataFrame(X_train_imp,columns=final_cols)
        f_test = pd.DataFrame(X_test_imp,columns=final_cols)
        f_train = f_train.fillna(0)
        f_test = f_test.fillna(0)
        
        save_used_cols = pd.DataFrame(f_train.columns,columns=['name'])
        save_used_cols.to_csv(os.path.join(path_results,'columns_used.csv'))

        if data_to_use=='image':
            f_train = f_train[cols_img]
            f_test = f_test[cols_img]
        elif data_to_use == 'clinical_no_img_scores':
            f_train = f_train[var[var['img']==0]['names']]
            f_test = f_test[var[var['img']==0]['names']]
        elif data_to_use == 'clin_no_scores_with_autofeats':
            f_train = f_train[list(var[var['img']==0]['names'])+list(cols_img)]
            f_test = f_test[list(var[var['img']==0]['names'])+list(cols_img)]
        
        #if und=='Y':
            #rus = RandomUnderSampler(random_state=1)
            #X_train_imp, y_train = rus.fit_resample(X_train_imp, y_train) 
        #y_train = np.array(y_train<=2,dtype='int32')
        #y_test = np.array(y_test<=2,dtype='int32')
        break 
    
        if task=='classification': 
            y_train = y_train<=2
            y_test = y_test<=2
            class_rfc=mt.Pipeline(True,'RFC',f_train,y_train,f_test,y_test,l,cv,mean_tprr,rfc_m,path_results,opt,und,final_cols)   
            class_svm=mt.Pipeline(True,'SVM',f_train,y_train,f_test,y_test,l,cv,mean_tprr,svm_m,path_results,opt,und,final_cols)   
            class_lr=mt.Pipeline(True,'LR',f_train,y_train,f_test,y_test,l,cv,mean_tprr,lr_m,path_results,opt,und,final_cols)
            class_nn=mt.Pipeline(True,'NN',f_train,y_train,f_test,y_test,l,cv,mean_tprr,nn_m,path_results,opt,und,final_cols) 
            class_xgb=mt.Pipeline(True,'XGB',f_train,y_train,f_test,y_test,l,cv,mean_tprr,xgb_m,path_results,opt,und,final_cols)  
        else:
            class_rfc = mt_r.Pipeline(True,'RFC',f_train,y_train,f_test,y_test,l,cv,mean_tprr,rfc_m,path_results,opt,und,final_cols)   
            class_svm = mt_r.Pipeline(True,'SVM',f_train,y_train,f_test,y_test,l,cv,mean_tprr,svm_m,path_results,opt,und,final_cols)   
            class_lr = mt_r.Pipeline(True,'LR',f_train,y_train,f_test,y_test,l,cv,mean_tprr,lr_m,path_results,opt,und,final_cols)
            class_nn = mt_r.Pipeline(True,'NN',f_train,y_train,f_test,y_test,l,cv,mean_tprr,nn_m,path_results,opt,und,final_cols) 
            class_xgb = mt_r.Pipeline(True,'XGB',f_train,y_train,f_test,y_test,l,cv,mean_tprr,xgb_m,path_results,opt,und,final_cols) 
        
        end_pipeline = time.time()
        print("Total time to process iteration: ",end_pipeline - start_pipeline)
                                                                  
    final_m=[rfc_m,svm_m,lr_m,xgb_m,nn_m]
    final_m=[x for x in final_m if x.run != False]
    names=[class_rfc.name,class_svm.name,class_lr.name,class_xgb.name,class_nn.name]
    names=[x for x in names if x != 'NONE'] 
    if task=='classification':
        mt.Print_Results_Excel(final_m,splits,names,path_results,l,data_to_use)
    else:
        mt_r.Print_Results_Excel(final_m,splits,names,path_results,l,data_to_use)
            
    #ut.Save_fpr_tpr(path_results,names,final_m)
    


