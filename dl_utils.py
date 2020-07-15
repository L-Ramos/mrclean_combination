# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:53:24 2020

@author: laramos
"""
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras as kr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Dropout,BatchNormalization, Conv1D,Conv2D, Conv3D, MaxPooling1D,MaxPooling2D, MaxPooling3D, GlobalAveragePooling1D, Flatten, LeakyReLU,AveragePooling3D, Input
  
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from sklearn.metrics import roc_auc_score
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import xlwt  
from scipy.stats import randint as sp_randint
import scipy as sp
import glob
import re
import pandas as pd

import utils as ut



class Measures:       
    def __init__(self,splits):
        
        self.auc=np.zeros(splits)
    
        self.f1=np.zeros(splits)
        self.sens=np.zeros(splits)
        self.spec=np.zeros(splits)
        self.ppv=np.zeros(splits)
        self.npv=np.zeros(splits)
        
        self.fp=np.zeros(splits)
        self.fn=np.zeros(splits)
        self.tp=np.zeros(splits)
        self.tn=np.zeros(splits)
        
        self.history = list()

        self.run=False

        self.preds=list()

# def write_image_doesnt_work(frame_ref,data_path,dict_imgs,train_or_test):
    
#     for id_img in frame_ref['ID'].unique():
#         print(id_img)
#         if frame_ref[frame_ref['ID']==id_img]['mRS'].iloc[0]<=2:
#             good_path = os.path.join(data_path,train_or_test,r"mrs\good",id_img+".npy")
#             x = sitk.GetArrayFromImage(sitk.ReadImage(dict_imgs[id_img]))
#             np.save(good_path,x)
#         elif frame_ref[frame_ref['ID']==id_img]['mRS'].iloc[0]>2:
#             good_path = os.path.join(data_path,train_or_test,r"mrs\not_good",id_img+".npy")
#             x = sitk.GetArrayFromImage(sitk.ReadImage(dict_imgs[id_img]))
#             np.save(good_path,x)
#         if frame_ref[frame_ref['ID']==id_img]['TICI'].iloc[0]>=3:
#             good_path = os.path.join(data_path,train_or_test,r"tici\good",id_img+".npy")
#             x = sitk.GetArrayFromImage(sitk.ReadImage(dict_imgs[id_img]))
#             np.save(good_path,x)
#         elif frame_ref[frame_ref['ID']==id_img]['TICI'].iloc[0]<3:
#             good_path = os.path.join(data_path,train_or_test,r"tici\not_good",id_img+".npy")
#             x = sitk.GetArrayFromImage(sitk.ReadImage(dict_imgs[id_img]))
#             x = np.clip(x,0,500)
#             np.save(good_path,x)
            
def write_image_dl(frame_ref,data_path,dict_imgs):       
    for id_img in frame_ref['ID'].unique():
        print(id_img)
        good_path = os.path.join(data_path,id_img+".npy")
        x = sitk.GetArrayFromImage(sitk.ReadImage(dict_imgs[id_img]))
        x = np.clip(x,0,500)
        x = np.swapaxes(x,0,2)
        x = np.swapaxes(x,0,1)        
        np.save(good_path,x)  

        frame_ref.to_csv(os.path.join(data_path,"labels.csv"),index=False)
    
def remove_failed_file(frame,folder_path,id_col):
    #removes the features from images that have a folder but failed (failed was defined by visual inspection)
 
    comb_ids = list(frame[id_col])
    
    failed_ids = np.load(os.path.join(folder_path,'failed_img_ids.npy'))
            
    all_ids = [i for i in comb_ids if i not in failed_ids]    
    frame_failed = frame[~frame[id_col].isin(all_ids)]
    frame = frame[frame[id_col].isin(all_ids)]
    return(frame,frame_failed)

def clean_fix_data(data_path):
    
    #reading data
    
    #folder_path = r"/home/ubuntu"
    
    folder_path = r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\data"
    
    path_data = os.path.join(folder_path,"complete_data_part2.csv")
    
    path_variables = os.path.join(folder_path,"Baseline_contscore_new_review.csv")
    print("Processing clinical data")
    frame,cols_o,var,data,Y_mrs,Y_tici,vals_mask,miss,subj,mask_cont,mask_cat = ut.Clean_Data(path_data, path_variables)
    
    #frame['mRS'] = np.array(Y_mrs<=2,dtype='int16')
     
    
    frame['mRS'] = np.array(Y_mrs)
    frame['TICI'] = Y_tici
    frame['ID'] = subj
    print("Selecting relevant labels")
    frame = frame[['ID','mRS','occlside_c','TICI']]
    
    #we can still crop the top of the head in the scans because that is mostly air
    print("Finding images")
    image_list_1  = glob.glob(r"L:\basic\Personal Archive\L\laramos\Disk_E\MrClean_part1\R****\resampled_raw.mha") 
    image_list_2  = glob.glob(r"L:\basic\Personal Archive\L\laramos\Disk_E\MrClean_part2\R****\resampled_raw.mha") 
    image_list = image_list_1 + image_list_2
        
    
        
    subj_imgs = list()
    dict_imgs = dict()
    print("Extracting Ids")
    for i in range(0,len(image_list)):
        id_img = re.search('R[0-9][0-9][0-9][0-9]',image_list[i])[0]        
        subj_imgs.append(id_img)
        dict_imgs[id_img] = image_list[i]
        
    frame,f_failed = remove_failed_file(frame,folder_path,id_col='ID')
    
    
    frame_img = pd.DataFrame((subj_imgs),columns=['ID'])
    frame = frame.merge(frame_img,on='ID')    
    print("Saving to the same folder")
    write_image_dl(frame,data_path,dict_imgs)
    
    frame.to_csv(os.path.join(data_path,"labels.csv"),index=False)
    print("Done")
    
  
    
    
class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs, data, image_path, cols_use, inc_clin = False, label_name='mRS',
                 batch_size=16, dim=(217, 181,30),
                 n_channels=1, n_classes=2, shuffle=True,conv_type = '3D'):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.data = data
        self.image_path = image_path
        self.label_name  = label_name
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.conv_type = conv_type
        self.feats = cols_use
        self.inc_clin = inc_clin

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.inc_clin:
            y = self._generate_y(list_IDs_temp)
            z = self._generate_z(list_IDs_temp)
            return (X,z),y
        else:
            y = self._generate_y(list_IDs_temp)            
            return X, y
            #return np.concatenate((X,np.flip(X,axis=2)),axis=0),np.concatenate((y,y),axis=0)
            
    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        if self.conv_type=='3D':
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
                    # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                X[i,:,:,:,0] = self._load_image(os.path.join(self.image_path, ID+".npy"))            
        else:
            X = np.empty((self.batch_size, *self.dim))
                    # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                X[i,:,:,:] = self._load_image(os.path.join(self.image_path, ID+".npy"))  

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size), dtype=int)

        # Generate data

        for i, ID in enumerate(list_IDs_temp):
            
            y[i] = self.data[self.data['s_id']==ID][self.label_name].iloc[0]
      
        return tf.keras.utils.to_categorical(y, self.n_classes)
    
    def _generate_z(self, list_IDs_temp):
        """Generates data containing batch_size clinical data
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        z = np.empty((self.batch_size,len(self.feats),1), dtype=int)

        # Generate data

        for i, ID in enumerate(list_IDs_temp):
           
            z[i,:,0] = self.data[self.data['s_id']==ID][self.feats]
      
        return z
    
    def _load_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = np.load(image_path)       
        img = img / 500
        return img
    
def get_metrics():
    return([
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),])

def cnn_model():
        
    inp_img = Input(shape=(217,181,30,1))
    
    conv1 = Conv3D(filters=16, kernel_size=(3,3,3), activation='relu', name="inp/conv1")(inp_img)
    avg_pool1 = AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)(conv1)
    dp1 = Dropout(0.3)(avg_pool1)
    
    conv2 = Conv3D(filters=16, kernel_size=(3,3,3), activation='relu', name="conv2")(dp1)
    avg_pool2 = AveragePooling3D(pool_size=(2, 2, 1), strides=None, padding='valid', data_format=None)(conv2)
    dp2 = Dropout(0.3)(avg_pool2)
    
    conv3 = Conv3D(filters=32, kernel_size=(3,3,3), activation='relu', name="conv3")(dp2)
    avg_pool3 = AveragePooling3D(pool_size=(2, 2, 1), strides=None, padding='valid', data_format=None)(conv3)
    dp3 = Dropout(0.3)(avg_pool3)
    
    conv4 = Conv3D(filters=32, kernel_size=(3,3,3), activation='relu', name="conv4")(dp3)
    avg_pool4 = AveragePooling3D(pool_size=(2, 2, 1), strides=None, padding='valid', data_format=None)(conv4)
    dp4 = Dropout(0.3)(avg_pool4)
    
    conv5 = Conv3D(filters=64, kernel_size=(3,3,3), activation='relu', name="conv5")(dp4)
    avg_pool5 = AveragePooling3D(pool_size=(2, 2, 1), strides=None, padding='valid', data_format=None)(conv5)
    dp5 = Dropout(0.3)(avg_pool5)
        
    flat = Flatten()(dp5)
    dense1 = Dense(256, activation='relu')(flat)
    dp6 = Dropout(0.3)(dense1)
    dense2 = Dense(2, activation='softmax')(dp6)
    
    model = Model(inp_img, dense2)
    
    model.summary()
        
    conv_type = '3D'
    return model,conv_type

def cnn_model_with_clinical():
        
    inp_img = Input(shape=(217,181,30,1))
    
    conv1 = Conv3D(filters=16, kernel_size=(3,3,3), activation='relu', name="inp/conv1")(inp_img)
    avg_pool1 = AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)(conv1)
    dp1 = Dropout(0.3)(avg_pool1)
    
    conv2 = Conv3D(filters=16, kernel_size=(3,3,3), activation='relu', name="conv2")(dp1)
    avg_pool2 = AveragePooling3D(pool_size=(2, 2, 1), strides=None, padding='valid', data_format=None)(conv2)
    dp2 = Dropout(0.3)(avg_pool2)
    
    conv3 = Conv3D(filters=32, kernel_size=(3,3,3), activation='relu', name="conv3")(dp2)
    avg_pool3 = AveragePooling3D(pool_size=(2, 2, 1), strides=None, padding='valid', data_format=None)(conv3)
    dp3 = Dropout(0.3)(avg_pool3)
    
    conv4 = Conv3D(filters=32, kernel_size=(3,3,3), activation='relu', name="conv4")(dp3)
    avg_pool4 = AveragePooling3D(pool_size=(2, 2, 1), strides=None, padding='valid', data_format=None)(conv4)
    dp4 = Dropout(0.3)(avg_pool4)
    
    conv5 = Conv3D(filters=64, kernel_size=(3,3,3), activation='relu', name="conv5")(dp4)
    avg_pool5 = AveragePooling3D(pool_size=(2, 2, 1), strides=None, padding='valid', data_format=None)(conv5)
    dp5 = Dropout(0.3)(avg_pool5)
        
    flat = Flatten()(dp5)
    dense1 = Dense(256, activation='relu')(flat)
    dp6 = Dropout(0.3)(dense1)
    dense2 = Dense(2, activation='softmax')(dp6)
    
    model = Model(inp_img, dense2)
    
    model.summary()
    conv_type = '3D'
    return model,conv_type



def get_weights_balance(frame_train,label_name):
    #as suggested in the tensorflow website
    sample_size = frame_train.shape[0]
    tot_pos = np.sum(frame_train[label_name])
    tot_neg = sample_size - tot_pos
    w_0 = (1 / tot_neg)*(sample_size)/2.0 
    w_1 = (1 / tot_pos)*(sample_size)/2.0
    return [w_0, w_1]


def vgg_16(input_shape = (64, 64, 3),input_shape_clin = (58), classes=6, inc_clin=False):

    X_input = Input(input_shape)
    clin = Input(input_shape_clin)
    
    X = (Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))(X_input)
    X = (Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))(X)
    X = (MaxPooling2D(pool_size=(2,2),strides=(2,2)))(X)
    X = (Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))(X)
    X = (Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))(X)
    X = (MaxPooling2D(pool_size=(2,2),strides=(2,2)))(X)
    X = (Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))(X)
    X = (Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))(X)
    X = (Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))(X)
    X = (MaxPooling2D(pool_size=(2,2),strides=(2,2)))(X)
    X = (Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))(X)
    X = (Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))(X)
    X = (Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))(X)
    X = (MaxPooling2D(pool_size=(2,2),strides=(2,2)))(X)
    X = (Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))(X)
    X = (Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))(X)
    X = (Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))(X)
    X = (MaxPooling2D(pool_size=(2,2),strides=(2,2)))(X)
    
    X = (Flatten())(X)
    c = Flatten()(clin)
    if inc_clin:
        combined = Concatenate()([X, c])
        X = (Dense(units=4096,activation="relu"))(combined)
        X = (Dense(units=4096,activation="relu"))(X)
        X = (Dense(units=2, activation="softmax"))(X)
        model = Model(inputs = [X_input,clin], outputs = X, name='vgg16')
    else:
        X = (Dense(units=4096,activation="relu"))(X)
        X = (Dense(units=4096,activation="relu"))(X)
        X = (Dense(units=2, activation="softmax"))(X)
        model = Model(inputs = X_input, outputs = X, name='vgg16')
    
    model.summary()
    conv_type = '2D'
    return model,conv_type





def Mean_Confidence_Interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.nanmean(a), sp.stats.sem(a,nan_policy = 'omit')
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def Print_Results_Excel_DL(m,names,path_results,l):    
    colors=['darkorange','blue','green','black','yellow']
    book = xlwt.Workbook(encoding="utf-8")    
    sheet1 = book.add_sheet("Sheet 1")
    #path_results_txt=path_results+path_results[2:len(path_results)-2]+str(l)+".xls"
    path_results_txt = os.path.join(path_results,('results'+str(l)+".xls"))
    sheet1.write(0, 0, "Methods")
    sheet1.write(0, 1, "AUC 95% CI ")
    sheet1.write(0, 2, "Sensitivity ")
    sheet1.write(0, 3, "Specificity")
    sheet1.write(0, 4, "PPV")
    sheet1.write(0, 5, "NPV")
    sheet1.write(0, 6, "FP")
    sheet1.write(0, 7, "FN")
    sheet1.write(0, 8, "TP")
    sheet1.write(0, 9, "TN")

    for i in range(0,len(names)):        
        print(i,names[i])
        sheet1.write(i+1,0,(names[i])) 
        sheet1.write(i+1,1,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].auc.reshape(-1)))))                      
        sheet1.write(i+1,2,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].sens.reshape(-1)))))              
        sheet1.write(i+1,3,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].spec.reshape(-1)))))              
        sheet1.write(i+1,4,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].ppv.reshape(-1)))))              
        sheet1.write(i+1,5,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].npv.reshape(-1)))))              
        sheet1.write(i+1,6,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].fp.reshape(-1)))))              
        sheet1.write(i+1,7,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].fn.reshape(-1)))))              
        sheet1.write(i+1,8,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].tp.reshape(-1)))))              
        sheet1.write(i+1,9,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].tn.reshape(-1)))))              
        #sheet1.write(i+1,3,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_f1_score.reshape(-1)))))                                            

        #np.save(file=os.path.join(path_results,('AUCs_'+names[i]+'.npy')),arr=m[i].clf_auc)
        #np.save(file=path_results+'Thresholds_'+names[i]+'.npy',arr=m[i].clf_thresholds)
        #mean_tpr=m[i].mean_tpr
        #mean_tpr /= splits
        #mean_tpr[-1] = 1.0
        #frac_pos_rfc  /= skf.get_n_splits(X, Y)
        #mean_fpr = np.linspace(0, 1, 100) 
        #mean_auc_rfc = auc(mean_fpr, mean_tpr)
        #plt.plot(mean_fpr, mean_tpr, color=colors[i],lw=2, label=names[i]+' (area = %0.2f)' % mean_auc_rfc)
        #plt.legend(loc="lower right")
        #np.save(file=os.path.join(path_results,('tpr_'+names[i]+'.npy')),arr=mean_tpr)
        #np.save(file=os.path.join(path_results,('fpr_'+names[i]+'.npy')),arr=mean_fpr)
        #if names[i]=='RFC':
        #    np.save(file=os.path.join(path_results,('Feat_Importance'+names[i]+'.npy')),arr=m[i].feat_imp)
    book.save(path_results_txt)        
        
def plot_figure(history,model_type,results_model,fold):
    plt.figure()
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('model auc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #plt.show()
    plt.savefig(os.path.join(results_model,model_type+"auc_train_validation_"+str(fold))+'.pdf')    