# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:31:49 2020

@author: laramos
"""
#pre processing
import pickle
import dl_utils as dl_ut
from tensorflow import keras

#%% For rpeparing the data for DL, copies the imag edata to one single folder
dl_ut.clean_fix_data("F:\image_data_raw")

#%%#Getting the features from a given layer                

#image_path = (os.path.join(img_data_path, list_IDs[1]+".npy"))
#img = np.load(image_path)       
# img = img / 500
# X[1,:,:,:] = img  
#X = np.array(X,dtype='float32')

model = keras.models.load_model(r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\DL_results\resnet50_mrsFalse\resnet504.h5")

intermediate_layer_model = Model(inputs=model.input,
        
                                 outputs=model.get_layer('flatten_22').output)
intermediate_output = intermediate_layer_model(X)
#%% finding best threshold for probabilities
fpr, tpr, thresholds = roc_curve(y_test[:], preds[:,1])
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]    

#plots history
ids = np.load(r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\DL_results\resnet32_mrsFalse\test_ids_0.npy",allow_pickle=True)
for i in ids:
    img = np.load(os.path.join(img_data_path,ids[i]+'.npy'))
    img = img / 500
    feats = intermediate_layer_model(img) 
#y_tr = tf.keras.utils.to_categorical(y, 2)
#%%
#loading measures and saving as excel (computes CI)

m=list()
model_list = ['resnet32','resnet50']
results_path_list = list()
results_path_list.append(r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\DL_results\resnet32_mrsTrue\measures_resnet32")
results_path_list.append(r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\DL_results\resnet50_mrsTrue\measures_resnet50")

for model in results_path_list:        
    with open(model, 'rb') as handle:
        m.append(pickle.load(handle))
    
dl_ut.Print_Results_Excel_DL(m,model_list,'',0)    
 
#testing augmentation
#%%

import numpy as np
import matplotlib.pyplot as plt

x = np.load(r"F:\image_data\R0001.npy")
x = x/500

plt.imshow(x[:,:,10],cmap='bone')

x_1 = np.flip(x,axis=1)

plt.imshow(x_1[:,:,10],cmap='bone')

y = tf.keras.utils.to_categorical(frame_train['mrs'], 2)

y_2 = np.concatenate((y,y),axis=0)

test_generator = dt_ut.DataGenerator(list(frame_train['s_id']), frame_train, img_data_path,cols_use, inc_clin=use_clin,
                                       label_name = label_name, batch_size=1, shuffle=False, conv_type = conv_type)    


 
y_train = np.array(frame_test[label_name],dtype='int16')
meas.auc[fold] = roc_auc_score(y_train[:],preds[:,1])        



#testing read torch save keras/tf

import onnx
from tensorflow.keras.models import load_model

pytorch_model = '/path/to/pytorch/model'
keras_output = '/path/to/converted/keras/model.hdf5'
onnx.convert(pytorch_model, keras_output)
model = load_model(keras_output)
preds = model.predict(x)
