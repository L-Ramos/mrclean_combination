cols_use# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 16:48:00 2019

@author: laramos
"""

import glob
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
import subprocess
import os

#%% Resampling and saving
image_list_1  = glob.glob(r"L:\basic\Personal Archive\L\laramos\Disk_E\MrClean_part1\R****\skull_stripped_scan.mhd")
image_list_2  = glob.glob(r"L:\basic\Personal Archive\L\laramos\Disk_E\MrClean_part2\R****\skull_stripped_scan.mhd")
image_list = image_list_1 + image_list_2 

for i in range(0,len(image_list)):
    path_img = image_list[i]
    h,t = os.path.split(path_img)
    print(h,i)
    path_out = h+'\\resampled_raw.mha'
    command = "c3d \""+path_img + "\" -interpolation NearestNeighbor -resample 256x256x30 -o \"" +path_out+"\""    
    print(subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read())

#%% cropping the brain

image_list_1  = glob.glob(r"L:\basic\Personal Archive\L\laramos\Disk_E\MrClean_part1\R****\resampled_raw.mha")
image_list_2  = glob.glob(r"L:\basic\Personal Archive\L\laramos\Disk_E\MrClean_part2\R****\resampled_raw.mha")
image_list = image_list_1 + image_list_2 

min_x = list()
max_x = list()
dist_x = list()
min_y = list()
max_y = list()
dist_y = list()

for i in range(0,len(image_list)):
    print(i, len(image_list))
    path_img = image_list[i]
    #img_path = r"L:\basic\Personal Archive\L\laramos\Disk_E\MrClean_part2\R0322\resampled_raw.mha"    
    img = sitk.ReadImage(path_img)
    img = sitk.GetArrayFromImage(img)

    x = np.where((np.sum(np.sum(img,axis=1),0)>0)==True)
    y = np.where((np.sum(np.sum(img,axis=2),0)>0)==True)
    
    min_x.append(x[0][0])
    max_x.append(x[0][-1])
    dist_x.append(x[0][-1]-x[0][0])
    min_y.append(y[0][0])
    max_y.append(y[0][-1])   
    dist_y.append(y[0][-1]-y[0][0])
    
print(np.max(dist_x),np.max(dist_y))    

#%%cropping    

min_x_crop = np.min(min_x)
max_x_crop = np.max(max_x)
min_y_crop = np.min(min_y)
max_y_crop = np.max(max_y)

    
for i in range(0,len(image_list)):
    path_img = image_list[i]
    #img_path = r"L:\basic\Personal Archive\L\laramos\Disk_E\MrClean_part2\R0322\resampled_raw.mha"    
    img = sitk.ReadImage(path_img)
    img = sitk.GetArrayFromImage(img)
    img_save = img[:,min_y_crop:max_y_crop,min_x_crop:max_x_crop]
    h,t = os.path.split(path_img)
    print(h,i)
    path_out = h+'\\cropped_resampled_raw.mha'
    
    sitk.WriteImage(img_save,path_out)
    


#img_plot = img[:,y[0][0]:y[0][-1],x[0][0]:x[0][-1]]
plt.imshow(img_save[15,:,:])
