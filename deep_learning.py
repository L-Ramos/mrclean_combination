# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:09:10 2019

@author: laramos
"""
import glob
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
import pickle
import re
import subprocess
import os
import pandas as pd

#from imblearn.under_sampling import RandomUnderSampler
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
import random
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import optimizers  
import tensorflow as tf
import random as python_random

from def_resnet import ResNet50,ResNet50_short,Resnet32,Resnet32_3d
from sklearn.metrics import roc_auc_score
import utils as ut
import dl_utils as dt_ut
from sklearn.metrics import roc_auc_score,f1_score,confusion_matrix,roc_curve
import pickle


s = 42
np.random.seed(s)
python_random.seed(s)
tf.random.set_seed(s)

def fix_clin_data(data_path,folder_path):
    
    
        frame = pd.read_csv(os.path.join(data_path,"labels.csv"))
        frame = frame.dropna()
        
        frame['occlside_c'] = (frame['occlside_c']>=1).astype('int16')
        
        frame_train,frame_test = train_test_split(frame, test_size=0.15, random_state = s)
        
        frame_train,frame_val = train_test_split(frame_train, test_size=0.10, random_state = s)
                
            
        path_data = os.path.join(folder_path,"complete_data_part2.csv")
        
        path_variables = os.path.join(folder_path,"Baseline_contscore_new_review.csv")
        
        _,cols_o,var,data,Y_mrs,Y_tici,vals_mask,miss,subj,mask_cont,mask_cat = ut.Clean_Data(path_data, path_variables)
        
        frame_clin = pd.DataFrame(data,columns=cols_o)
        frame_clin['ID'] = subj
        cols_merge = ['ID','mRS','TICI']
                
        X_train_imp = pd.read_csv('train.csv')
        X_val_imp = pd.read_csv('val.csv')
        X_test_imp = pd.read_csv('test.csv')
        
        X_train_imp['mRS'] = (X_train_imp['mRS']<=2).astype('int16')
        X_val_imp['mRS'] = (X_val_imp['mRS']<=2).astype('int16')
        X_test_imp['mRS'] = (X_test_imp['mRS']<=2).astype('int16')
        
        X_train_imp['TICI'] = (X_train_imp['TICI']>=3).astype('int16')
        X_val_imp['TICI'] = (X_val_imp['TICI']>=3).astype('int16')
        X_test_imp['TICI'] = (X_test_imp['TICI']>=3).astype('int16')
                
        cols_o = X_train_imp.columns
        
        vals_mask = ut.clean_mask(vals_mask,cols_o)
        
        X_train_imp,X_val_imp, X_test_imp,cols_recoded = ut.Change_One_Hot_DL(X_train_imp,X_val_imp, X_test_imp,vals_mask)
                 
        scaler = ColumnTransformer([('norm1', MinMaxScaler(),mask_cont)], remainder='passthrough')
        
        X_train_imp = pd.DataFrame(X_train_imp,columns=cols_recoded)
        X_val_imp = pd.DataFrame(X_val_imp,columns=cols_recoded)
        X_test_imp = pd.DataFrame(X_test_imp,columns=cols_recoded)
        

         
        scaler = scaler.fit(X_train_imp)    
        X_train_imp = scaler.transform(X_train_imp)
        X_val_imp = scaler.transform(X_val_imp)
        X_test_imp = scaler.transform(X_test_imp)
                
        frame_train = pd.DataFrame(X_train_imp,columns=final_cols)
        frame_train = frame_train.drop_duplicates()
        frame_val = pd.DataFrame(X_val_imp,columns=cols_recoded)
        frame_val = frame_val.drop_duplicates()
        frame_test = pd.DataFrame(X_test_imp,columns=cols_recoded)
        frame_test = frame_test.drop_duplicates()
        
        cols_use = [c for c in cols_recoded if c not in cols_merge]
    
        return(frame_train,frame_val,frame_test,cols_use)

def fix_clin_data_imputed(data_path,fold):
    
    X_train = pd.read_csv(os.path.join(data_path,"imp_data_train"+str(fold)+".csv"))
    
    y_train = pd.read_csv(os.path.join(data_path,"imp_y_train"+str(fold)+".csv"))
    l_names = list(y_train.columns)
    for var in y_train:
        X_train[var] = y_train[var]
        
    frame_train,frame_val = train_test_split(X_train,test_size=0.15, random_state = s)
    
    frame_test = pd.read_csv(os.path.join(data_path,"imp_data_test"+str(fold)+".csv"))
    y_test = pd.read_csv(os.path.join(data_path,"imp_y_test"+str(fold)+".csv"))

    for var in y_test:
        frame_test[var] = y_test[var]
    del y_train, y_test    
    
    #vals_mask=['occlsegment_c_cbs','occlside_c'] 
    vals_mask=['occlsegment_c_cbs'] 
        
    frame_train,frame_val, frame_test,cols_recoded = ut.Change_One_Hot_DL(frame_train,frame_val, frame_test,vals_mask)

                     
    var = pd.read_csv(os.path.join(data_path,"Baseline_contscore_new_review.csv"))
    
    mask_cont = list(var[var['type']=='cont']['names'])
    mask_cat = list(var[var['type']=='cat']['names'])
    
    scaler = ColumnTransformer([('norm1', MinMaxScaler(),mask_cont)], remainder='passthrough')
    
    frame_train = pd.DataFrame(frame_train,columns=cols_recoded)
    frame_val = pd.DataFrame(frame_val,columns=cols_recoded)
    frame_test = pd.DataFrame(frame_test,columns=cols_recoded)
    
    mask_cat_recoded = [f for f in cols_recoded if f not in mask_cont]
    final_cols = list(mask_cont.copy())
    final_cols.extend(mask_cat_recoded)
     
    scaler = scaler.fit(frame_train)    
    frame_train = scaler.transform(frame_train)
    frame_val = scaler.transform(frame_val)
    frame_test = scaler.transform(frame_test)
    
    frame_train = pd.DataFrame(frame_train,columns=final_cols)
    frame_val = pd.DataFrame(frame_val,columns=final_cols)
    frame_test = pd.DataFrame(frame_test,columns=final_cols)
    
    frame_train['mrs'] = (frame_train['mrs']<=2).astype('int16')
    frame_val['mrs'] = (frame_val['mrs']<=2).astype('int16')
    frame_test['mrs'] = (frame_test['mrs']<=2).astype('int16')
    
    frame_train['posttici_c'] = (frame_train['posttici_c']>=3).astype('int16')
    frame_val['posttici_c'] = (frame_val['posttici_c']>=3).astype('int16')
    frame_test['posttici_c'] = (frame_test['posttici_c']>=3).astype('int16')
    l_names.append('s_id')
    
    cols_use = [f for f in frame_train.columns if f not in l_names]
    
    return(frame_train,frame_val,frame_test,cols_use)
    

img_data_path = "F:\image_data"

data_path = r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\data\imputed"

label_name = 'posttici_c'  #mrs

use_clin = True
batch_size = 8
epochs = 100
model_list = ['resnet32','resnet50','my3d','resnet32_3d']
input_shape = (217, 181, 30)
input_shape_clin = (58,1)
classes = 2
total_folds = 5

results_main = 'DL_results'
results_path_list = list()

for model_type in model_list:
    meas = dt_ut.Measures(total_folds)
    results_model = os.path.join(results_main,(model_type+"_" + label_name+str(use_clin)))
    if not os.path.exists(results_model):
            os.makedirs(results_model)
    results_path_list.append(results_model)            
    for fold in range(0,total_folds):
                      
        frame_train,frame_val,frame_test,cols_use = fix_clin_data_imputed(data_path,fold)
            
        if model_type == 'vgg':
            model,conv_type = dt_ut.vgg_16(input_shape = input_shape,input_shape_clin = input_shape_clin, classes = classes,inc_clin = use_clin)
        elif model_type == 'resnet50': 
            model,conv_type = ResNet50_short(input_shape = input_shape ,input_shape_clin = input_shape_clin, classes = classes,inc_clin = use_clin)
        elif model_type == 'resnet32':
            model,conv_type = Resnet32(input_shape = input_shape,input_shape_clin = input_shape_clin, classes = classes,inc_clin = use_clin)
        elif model_type == 'resnet32_3d' :  
            model,conv_type = Resnet32_3d(input_shape = input_shape,input_shape_clin = input_shape_clin, classes = classes,inc_clin = use_clin)
        elif model_type == 'my3d':
            model,conv_type = dt_ut.cnn_model()
                
        training_generator = dt_ut.DataGenerator(list(frame_train['s_id']), frame_train, img_data_path,cols_use, inc_clin=use_clin,
                                           label_name = label_name, batch_size=batch_size, shuffle=True, conv_type = conv_type)
        
        validation_generator = dt_ut.DataGenerator(list(frame_val['s_id']), frame_val, img_data_path,cols_use, inc_clin=use_clin,
                                             label_name = label_name, batch_size=batch_size, shuffle=True, conv_type = conv_type)
        
        test_generator = dt_ut.DataGenerator(list(frame_test['s_id']), frame_test, img_data_path,cols_use, inc_clin=use_clin,
                                       label_name = label_name, batch_size=1, shuffle=False, conv_type = conv_type)    
        
        #opt = optimizers.RMSprop(decay=1e-6)
        #opt = optimizers.Adam(decay=1e-6)
        opt = optimizers.Adam(learning_rate = 0.00003,decay=1e-6)
        
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics = dt_ut.get_metrics())
        
        #alpha = tf.random.normal()
        #beta = tf.random.normal()
        #model.compile(optmizer='someOptimizer',loss=[loss1,loss2],loss_weights=[alpha,1-alpha])
        
        class_weights = dt_ut.get_weights_balance(frame_train,label_name)
        
        checkpoint = ModelCheckpoint(os.path.join(results_model,model_type+str(fold)+".h5"), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
        
        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
        
        history = model.fit(training_generator, 
                                      epochs=epochs, 
                                      validation_data = validation_generator,
                                      #callbacks=[early_stopping, learning_rate_scheduler],                              
                                      class_weight=class_weights,
                                      #workers=-1,
                                      verbose = 1,
                                      callbacks=[checkpoint,early])
        
        dt_ut.plot_figure(history,model_type,results_model,fold)
        
        preds = model.predict_generator(test_generator, workers=-1,verbose = 1)      
        
        y_test = np.array(frame_test[label_name],dtype='int16')
        meas.auc[fold] = roc_auc_score(y_test[:],preds[:,1])
        
        tn, fp, fn, tp =  confusion_matrix(y_test[:], preds[:,1].reshape(-1,1)>=0.5).ravel() 
        
        meas.tn[fold],meas.fp[fold],meas.fn[fold],meas.tp[fold] = tn, fp, fn, tp 
        
        meas.sens[fold]=tp/(tp+fn)
        meas.spec[fold]=tn/(tn+fp)
        meas.ppv[fold]=tp/(tp+fp)
        meas.npv[fold]=tn/(tn+fn) 
          
        meas.preds.append(preds)  
        #meas.history.append(history) 
        np.save(os.path.join(results_model,'test_ids_'+str(fold)+'.npy'),np.array(frame_test['s_id']))
        del model
    with open(os.path.join(results_model,"measures_"+model_type), 'wb') as handle:
        pickle.dump(meas,handle)
        
    #del model,meas,early,checkpoint,history


