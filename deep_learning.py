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
import re
import subprocess
import os
import pandas as pd
import data_preprocessing as dp
from imblearn.under_sampling import RandomUnderSampler
from numpy.random import seed

s = 42

seed(s)


#reading data

#path to the complete dataset    
path_data=r"\\amc.intra\users\L\laramos\home\Desktop\MrClean_Poor\data\data_complete.csv"
#path to the variables to the used
path_variables=r"\\amc.intra\users\L\laramos\home\Desktop\MrClean_Poor\data\Baseline_contscore_new.csv"

frame,cols_o,var,data,Y_mrs,Y_tici,data_img,vals_mask,miss,original_mrs,subj = dp.Clean_Data(path_data,path_variables)
frame = pd.DataFrame(data,columns=cols_o)
#frame['mRS'] = np.array(Y_mrs<=2,dtype='int16')
 #frame=pd.read_csv(path_data,sep=';',encoding = "latin",na_values=' ')
 
side = np.array(frame['occlside_c'])

frame['mRS'] = np.array(Y_mrs)
frame['TICI'] = Y_tici
frame['ID'] = subj

frame = frame[['ID','mRS','occlside_c','TICI']]

#we can still crop the top of the head in the scans because that is mostly air

image_list  = glob.glob(r"E:\MrClean_part1\R****\resampled.mha") 

#This is only for now, should do imputation to fix this
frame = frame.dropna()

#X = np.zeros(((frame.shape[0]),30,217,181),dtype="float32")
X=list()
subj_imgs = list()
labels = list()
labels_side = list()
labels_tici = list()
subj_list = list()
miss=list()
k=0
for i in range(0,len(image_list)):
    id_img = re.search('R[0-9][0-9][0-9][0-9]',image_list[i])[0]    
    subj_imgs.append(id_img)
    j=0
    while j<frame.shape[0]:
        if frame['ID'].iloc[j]==id_img:
            subj_list.append(frame['ID'].iloc[j])
            labels.append(frame['mRS'].iloc[j])
            labels_side.append(frame['occlside_c'].iloc[j])
            labels_tici.append(frame['TICI'].iloc[j])
            #X[k,:,:,:] = sitk.GetArrayFromImage(sitk.ReadImage(image_list[i]))
            X.append(sitk.GetArrayFromImage(sitk.ReadImage(image_list[i])))
            k+=1
            break
        j+=1
        

y_mrs = np.array(labels)
y_mrs = y_mrs <= 2

y_side = np.array(labels_side)
y_side = y_side>=1

y_tici = np.array(labels_tici)
y_tici = y_tici>=3


#Define here which label you want to use ------------
y = y_side
#label ---------------------------------------------

X = np.array(X,dtype = "float32")

v = np.max(X)
np.where(X==v)

v = np.min(X)
np.where(X==v)





# Undersampling the majority
rus = RandomUnderSampler(return_indices=True,random_state=42)
_, _, idx = rus.fit_sample(y.reshape(-1, 1), y)
X, y = X[idx], y[idx]



import keras.backend as K
import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Conv3D, MaxPooling1D, MaxPooling3D, GlobalAveragePooling1D, Flatten, LeakyReLU,AveragePooling3D, Input
from keras.layers.normalization import BatchNormalization
from keras import optimizers    
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from keras.models import Model, load_model



y_tr = keras.utils.to_categorical(y, 2)

X = np.expand_dims(X,axis=4)

X = np.swapaxes(X,1,3)
X = np.swapaxes(X,1,2)

X = np.clip(X,0,1000)

X=X/np.max(X)


X_train, X_test, y_train, y_test = train_test_split(X, y_tr, test_size=0.1, random_state = s,stratify=y_tr)

inp_img = Input(shape=(217,181,30,1))

conv1 = Conv3D(filters=16, kernel_size=(3,3,3), activation='relu', name="inp/conv1")(inp_img)
#model.add(Conv3D(filters=16, kernel_size=(3,3,3), activation='relu', input_shape=(30,217,181,1), name="conv1_1"))
avg_pool1 = AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)(conv1)
dp1 = Dropout(0.3)(avg_pool1)

conv2 = Conv3D(filters=16, kernel_size=(3,3,3), activation='relu', name="conv2")(dp1)
#model.add(Conv3D(filters=16, kernel_size=(3,3,3), activation='relu', name="conv2_2"))
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

model = Model(input=inp_img, output=[dense2])

print(model.summary())


#sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
rms = optimizers.RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])

epochs = 20

history = model.fit(X_train,
                      y_train,
                      #batch_size=int(X_tr.shape[0]/4),
                      batch_size=4,
                      shuffle=True,
                      epochs=epochs,
                      validation_split=0.10,                      
                      verbose=1)

model.save('occ_side.h5')



model = load_model('occ_side.h5')



out = Flatten()(model.layers[-5].output)
dense1 = Dense(256, activation='relu')(model.layers[-3].output)
dp = (Dropout(0.3))(dense1)
dense2 = (Dense(2, activation='softmax'))(dp)


model2 = Model(input=in_img, output=[dense2])
model2.summary()


pred=list()

for i in range(X_test.shape[0]):
    s = np.zeros((1,217,181,30,1))
    s[0,:,:,:,:] = X_test[i,:,:,:,:]
    pred.append(model.predict(s))

pred = np.array(pred)
acc = (accuracy_score(y_test[:,1], (pred[:,0,1] > 0.5)))
auc = (roc_auc_score(y_test[:,1], (pred[:,0,1] > 0.5)))
s = pred[:,0,:]
#history = model.fit(X,
#                      y_tr,
#                      #batch_size=int(X_tr.shape[0]/4),
#                      batch_size=4,
#                      shuffle=True,
#                      epochs=epochs,
#                      #validation_split=0.10,                      
#                      verbose=1)