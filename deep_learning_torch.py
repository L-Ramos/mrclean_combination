# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:09:10 2019

@author: laramos
"""

#"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"

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
from transform import transforms

#from imblearn.under_sampling import RandomUnderSampler
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
import random
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import random as python_random
from sklearn.metrics import roc_auc_score,f1_score,confusion_matrix,roc_curve
import pickle

from def_resnet_torch import ResNet18,ResNet34,Feedforward
from def_resnet_torch_3D import ResNet18_3D,ResNet34_3D
import utils as ut
import dl_utils_torch as dt_ut
from tqdm import tqdm
from dl_utils_torch import BrainSegmentationDataset as Dataset

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transform import transforms

s = 42
np.random.seed(s)
python_random.seed(s)
#tf.random.set_seed(s)


class Measures():
    def __init__(self):
        self.acc = list()
        self.spec = list()
        self.sens = list()
        self.auc_list = list()
        self.predictions = list()
        self.labels = list()
        self.auc = list()
        self.f1 = list()
        self.loss_train = list()
        self.loss_val = list()

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
    
def datasets(path_data,idx_samples,batch_size,data,label_name,transform,cols_use,conv_type,only_clin):
    
    train = Dataset(
        path_data = path_data,       
        idx_samples = idx_samples,
        batch_size = batch_size,
        data = data,
        label_name = label_name,
        transform = transform,
        cols_use = cols_use,
        conv_type = conv_type,
        only_clin = only_clin
    )
    return(train)



def data_loaders(path_data,idx_train_samples,idx_val_samples,idx_test_samples,batch_size,frame_train,frame_val,frame_test,label_name,cols_use,conv_type, only_clin):
    #transform = transforms(scale=0.05, angle=15, flip_prob=0.5)
    transform = transforms(scale=None, angle=None, flip_prob=None)
    dataset_train = datasets(path_data,idx_train_samples,batch_size,frame_train,label_name,transform,cols_use,conv_type,only_clin)
    dataset_validation = datasets(path_data,idx_val_samples,batch_size,frame_val,label_name,None,cols_use,conv_type,only_clin)
    dataset_test = datasets(path_data,idx_test_samples,batch_size,frame_test,label_name,None,cols_use,conv_type,only_clin)
    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        
    )    
    loader_validation = DataLoader(
        dataset_validation,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,        
    )  
    
    loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,        
    )  
        
    return loader_train ,loader_validation, loader_test

#%%

img_data_path = "F:\image_data_raw"

path_data = r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\data\imputed"

label_name = 'mrs'  #mrs

device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda:0')

use_clin = False

batch_size = [2,8,8]
epochs = 50
model_list = ['resnet18_3D','resnet18','resnet34']
#model_list = ['feed_forward','resnet18','resnet34']
input_shape_clin = (58,1)
classes = 2
total_folds = 1


results_main = 'DL_results'
results_path_list = list()

for c,model_type in enumerate(model_list):
    only_clin = False
    meas = dt_ut.Measures(total_folds)
    results_model = os.path.join(results_main,(model_type+"_" + label_name+str(use_clin)))
    if not os.path.exists(results_model):
            os.makedirs(results_model)
    results_path_list.append(results_model)
    meas = Measures()
    for fold in range(0,total_folds):
                      
        frame_train,frame_val,frame_test,cols_use = fix_clin_data_imputed(path_data,fold)
        if not use_clin:
            cols_use = None
            
            
        if model_type == 'resnet18':
            model = ResNet18(use_clin)
            conv_type = '2D'
        elif model_type == 'resnet34': 
            model = ResNet34(use_clin)
            conv_type = '2D'
        elif model_type == 'resnet18_3D': 
            model = ResNet18_3D(use_clin)
            conv_type = '3D'
        elif model_type == 'resnet34_3D': 
            model = ResNet34_3D(use_clin)
            conv_type = '3D'
        elif model_type == 'feed_forward': 
            model = Feedforward(58,40)
            conv_type = '3D'  
            only_clin = True

            
        training_generator, validation_generator, test_generator =  data_loaders(img_data_path,list(frame_train['s_id']),list(frame_val['s_id']),list(frame_test['s_id']),
                                                                                 batch_size[c],frame_train,frame_val,frame_test,label_name,cols_use,conv_type,only_clin)
                
        loaders = {"train": training_generator, "validation": validation_generator}
        
        model.to(device)
        dsc_loss = torch.nn.BCELoss()
        #dsc_loss = torch.nn.BCEWithLogitsLoss()
        #SGD, 0:00003, momentum 0.9
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
        #optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-6)

        loss_train_epoch = []
        loss_valid_epoch = []

        model.train()
        
        for e in range(0,epochs):
            
            print("Training epoch: ",e)
                
            for phase in ["train","validation"]:            
    
                    loss_train = []
                    loss_valid = []
                    
                    for i, data in tqdm(enumerate(loaders[phase])):                        
                        #torch.cuda.empty_cache()
                        if only_clin:
                             z, y_true = data
                             z, y_true = z.to(device), y_true.to(device)
                        else: 
                            if cols_use is not None:
                                x, y_true, z = data
                                x, y_true, z = x.to(device), y_true.to(device), z.to(device)
                            else:
                                x, y_true = data
                                x, y_true = x.to(device), y_true.to(device)
                        #print(phase,i)
                        
            
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(phase == "train"):
                            if only_clin:
                                y_pred = model(z)
                            else:
                                if cols_use is not None:
                                    y_pred = model(x,z)
                                else:
                                    y_pred = model(x)
                
                            loss = dsc_loss(y_pred, y_true)
                
                            if phase == "validation":
                                #print("validation!")
                                loss_valid.append(loss.item())
                                
                            if phase == "train":
                                loss_train.append(loss.item())
                                loss.backward()
                                optimizer.step()
                            #print(loss.item())
                    if phase == "train":
                        print("loss train: ",np.mean(loss_train))
                        loss_train_epoch.append(np.mean(loss_train))
                    else:
                        print("loss val: ",np.mean(loss_valid))
                        loss_valid_epoch.append(np.mean(loss_valid))
                    
        torch.save(model.state_dict(), os.path.join(results_model,'model_torch'+str(fold)+'.pt')) 
        
        preds = []        
        for i, data in enumerate(test_generator): 
            if only_clin:
                 z, y_true = data
                 z, y_true = z.to(device), y_true.to(device)
            else: 
                if cols_use is not None:
                    x, y_true, z = data
                    x, y_true, z = x.to(device), y_true.to(device), z.to(device)
                else:
                    x, y_true = data
                    x, y_true = x.to(device), y_true.to(device)
           
            if only_clin:
                y_pred = model(z)
            else:
                if cols_use is not None:
                    y_pred = model(x,z)
                else:
                    y_pred = model(x)
                
            y_pred = y_pred.detach().cpu().numpy()
            preds.append(y_pred) 
        
        preds = np.array(preds)    
        probas = torch.from_numpy(preds.astype(np.float32)) 
        probas = torch.sigmoid(probas)  
        probas = probas.detach().cpu().numpy()    
        y_true = np.array(frame_test[label_name]) 
        
        tn, fp, fn, tp = confusion_matrix(y_true[:], (probas[:,0] > 0.5)).ravel()          
        meas.sens.append(tp/(tp+fn))
        meas.spec.append(tn/(tn+fp))
        meas.acc.append(accuracy_score(y_true[:], (probas[:,0] > 0.5)))
        meas.f1.append(f1_score(y_true[:], (probas[:,0] > 0.5)))
        meas.auc.append(roc_auc_score(y_true,probas[:,0,0]))
        meas.labels.append(y_true)
        meas.predictions.append(probas)
        meas.loss_train.append(loss_train_epoch)
        meas.loss_val.append(loss_val_epoch)
        
        print("         ")
        print("Testing AUC:",meas.auc)
        print("         ")
        del model
        
    with open(os.path.join(results_model,'measures_torch.pkl'), 'wb') as f:
        pickle.dump(meas,f)


#%%Transfer Learning

from generate_model_transfer import generate_model,parse_opts

#path = r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\torch\MedicalNet_pytorch_files\pretrain\resnet_50_23dataset.pth"
path = r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\torch\MedicalNet_pytorch_files\pretrain\resnet_10.pth"



sets = parse_opts()
sets.model_depth = 10
sets.pretrain_path = path
sets.resume_path = path
sets.gpu_id = [device.index]

sets.target_type = "normal"
sets.phase = 'train'

checkpoint = torch.load(sets.resume_path)
    

model, _ = generate_model(sets)
model.load_state_dict(checkpoint['state_dict'],strict=False)



model.to(device)
dsc_loss = torch.nn.BCELoss()
#dsc_loss = torch.nn.BCEWithLogitsLoss()
#SGD, 0:00003, momentum 0.9
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01,momentum= 0.9)
#optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-6)

loss_train_epoch = []
loss_valid_epoch = []

model.train()

for e in range(0,epochs):
    
    print("Training epoch: ",e)
        
    for phase in ["train","validation"]:            

            loss_train = []
            loss_valid = []
            
            for i, data in tqdm(enumerate(loaders[phase])):                        
                #torch.cuda.empty_cache()
                if only_clin:
                     z, y_true = data
                     z, y_true = z.to(device), y_true.to(device)
                else: 
                    if cols_use is not None:
                        x, y_true, z = data
                        x, y_true, z = x.to(device), y_true.to(device), z.to(device)
                    else:
                        x, y_true = data
                        x, y_true = x.to(device), y_true.to(device)
                #print(phase,i)
                
    
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    if only_clin:
                        y_pred = model(z)
                    else:
                        if cols_use is not None:
                            y_pred = model(x,z)
                        else:
                            y_pred = model(x)
        
                    loss = dsc_loss(y_pred, y_true)
        
                    if phase == "validation":
                        #print("validation!")
                        loss_valid.append(loss.item())
                        
                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()
                    #print(loss.item())
            if phase == "train":
                print("loss train: ",np.mean(loss_train))
                loss_train_epoch.append(np.mean(loss_train))
            else:
                print("loss val: ",np.mean(loss_valid))
                loss_valid_epoch.append(np.mean(loss_valid))
            
torch.save(model.state_dict(), os.path.join(results_model,'model_torch'+str(fold)+'.pt')) 


    #     dt_ut.plot_figure(history,model_type,results_model,fold)
        
    #     preds = model.predict_generator(test_generator, workers=-1,verbose = 1)      
        
    #     y_test = np.array(frame_test[label_name],dtype='int16')
    #     meas.auc[fold] = roc_auc_score(y_test[:],preds[:,1])
        
    #     tn, fp, fn, tp =  confusion_matrix(y_test[:], preds[:,1].reshape(-1,1)>=0.5).ravel() 
        
    #     meas.tn[fold],meas.fp[fold],meas.fn[fold],meas.tp[fold] = tn, fp, fn, tp 
        
    #     meas.sens[fold]=tp/(tp+fn)
    #     meas.spec[fold]=tn/(tn+fp)
    #     meas.ppv[fold]=tp/(tp+fp)
    #     meas.npv[fold]=tn/(tn+fn) 
          
    #     meas.preds.append(preds)  
    #     #meas.history.append(history) 
    #     np.save(os.path.join(results_model,'test_ids_'+str(fold)+'.npy'),np.array(frame_test['s_id']))
    #     del model
    # with open(os.path.join(results_model,"measures_"+model_type), 'wb') as handle:
    #     pickle.dump(meas,handle)
        
    #del model,meas,early,checkpoint,history
#%% checking results
for ID in (frame_train['s_id']):
    s=np.load(os.path.join(img_data_path, ID+".npy"))
    if s.shape[0]!=256:
        print(ID,s.shape)
              
              
with open(os.path.join(results_model,'measures_torch.pkl'), 'rb') as f:
    s = pickle.load(f)
    
path = r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\torch\DL_results\feed_forward_mrsTrue\measures_torch.pkl"


with open(path, 'rb') as f:
    s = pickle.load(f)
    
#%% checking clinical data
from sklearn.linear_model import LogisticRegression    

clf = LogisticRegression(random_state=0).fit(frame_train[cols_use], frame_train[label_name])    
pred = clf.predict_proba(frame_test[cols_use])
roc_auc_score(frame_test[label_name],pred[:,1])
