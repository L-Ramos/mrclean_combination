# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:20:03 2020

@author: laramos
"""

# -*- coding: utf-8 -*-
"""
This code is for finding the CT/CTA with thinnest slices, skull strip and register to an atlas


TODO:

@author: laramos
"""

import SimpleITK as sitk
import glob
import numpy as np
from inference import segment_brain
import os
from tqdm import tqdm
import re
from utils import read_image, load_model, save_image
import torch
import time
from subprocess import call
import subprocess
from scipy.ndimage.morphology import binary_fill_holes
import pydicom
import cv2

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

general_path = glob.glob(r'L:\basic\divi\Projects\mrclean\MRCLEAN-Registry\Imaging_Part2_Sorted\CTA_BL\A_ASSESSEDBYCORELAB\R**',recursive=True)

#general_path = glob.glob(r'L:\basic\divi\Projects\mrclean\MRCLEAN-Registry\Imaging_Part1_Sorted\CTA_BL\R*')
model_path = r"H:\Desktop\Combination_MrCLean\imaging\bm_param.tar"
path_results = r"E:\MrClean_part2"
fixed_image_path = r"H:\\Desktop\Combination_MrCLean\imaging\Registration\Atlas\skull_stripped_atlas.mhd"
elastix_location = "H:\\Desktop\\Combination_MrCLean\\imaging\\elastix.exe"

param_rigid = r"H:\\Desktop\\Combination_MrCLean\\imaging\\Parameters_Rigid_Manon.txt"
param_affine = r"H:\\Desktop\\Combination_MrCLean\\imaging\\Parameters_Affine.txt"

model = load_model(model_path, device)



#for i in tqdm(range(0,len(general_path))):
#    temp_gen_path = general_path[i]    
#    move = os.listdir(temp_gen_path)
#    if len(move)>1:
#        print(temp_gen_path)

#%%
from tqdm import tqdm
num_files = list()
path_files = list()

not_files_i_want = ['.dat','.txt','.dll','.exe','.dat','.xml','.bat']
#Searching for the files, since the folders are not always properly structured
for i in tqdm(range(0,len(general_path))):
    temp_gen_path = general_path[i]    
    found=False
    list_move = os.listdir(temp_gen_path)
    
    for move in list_move:    
        _, file_extension = os.path.splitext(temp_gen_path+"\\"+move)
        if len(move)>0 and file_extension!='' and file_extension not in not_files_i_want:
            direcs = os.listdir(temp_gen_path+"\\"+move)    
            _, file_extension = os.path.splitext(direcs[0])
            if file_extension=='.dcm':
                path_files.append(temp_gen_path+"\\"+move) 
            else:
                if file_extension!='.dcm' and file_extension not in not_files_i_want:
        #This means theres only one folder inside this directory
                    complete_path = temp_gen_path+"\\"+move
                    while len(direcs)==1:
                        complete_path = complete_path+"\\"+direcs[0]
                        direcs = os.listdir(complete_path)
                        #temp_gen_path = temp_gen_path+"\\"+next_dir
                        #direcs = os.listdir(temp_gen_path)    
                    #This means this is the directory with the imanges
                    if len(direcs)>=10:
                        filename, file_extension = os.path.splitext(direcs[0])
                        path_files.append(complete_path)  
                    #This is a directory with multiple scans, so there are a couple of folders here        
                    else:
                        temp_gen_path_save = temp_gen_path+"\\"+move
                        for file in direcs:
                            f_path = temp_gen_path_save+"\\"+file
                            _,file_extension = os.path.splitext(f_path)
                            if file_extension=='.dcm':
                            #print(f_path)
                                path_files.append(f_path)     




path_files = sorted(path_files)        

np.save(r"E:\MrClean_part2\path_files_2.npy",path_files) 

#%%
from collections import defaultdict

path_files = sorted(np.load(r"E:\MrClean_part1\path_files.npy"))

path_dict = defaultdict(list) 

already_done = glob.glob(r"E:\MrClean_part1\R*")

# I'm also going to remove the keys that are already in the folder, that means they have already been registered and crashed somewhere before the end
id_list = list()
for f in (path_files):
     pat_id = re.search('R[0-9][0-9][0-9][0-9]',f)

     #id_list.append(pat_id)
     if not (any(pat_id[0] in mystring for mystring in already_done)):
         path_dict[pat_id.group(0)].append(f)    
     
#%%
k=list(path_dict.keys())
i=0
while k[i]!='R1271':
    del path_dict[k[i]]
    i=i+1    
     
        
                      
#%%        
#0-200 path files done   
#path_files = sorted(np.load(r"E:\MrClean_part2\path_files_2.npy",allow_pickle=True))
#all_files = path_files


#path_files = all_files[0:500]         
#path_files = all_files[500:1000]        

error_name = "E:\MrClean_part2\errors.npy"
#error_name = "E:\MrClean_part2\errors_3.npy"
#for file in path_files:  
error = list()         

thickness_dict = dict()

for key in path_dict:
    try:
        thinner = False
        pat_id = key
        
        save_path = path_results+'\\'+pat_id
        file_list  = path_dict[key]
        
        for file in file_list:
            print("Reading image: ",file)
            start = time.time()  
            #file = r"L:\basic\divi\Projects\mrclean\MRCLEAN-Registry\Imaging_Part1_Sorted\CTA_BL\R0001\12548115\1.3.6.1.4.1.40744.9.145328205372987504102346003904980144625"        
            #file = r"L:\basic\divi\Projects\mrclean\MRCLEAN-Registry\Imaging_Part1_Sorted\CTA_BL\R0004\32956159\1.3.6.1.4.1.40744.9.88757116754949946036156777226315937309"
                                   
            #file=file.replace('\\','/')
            
            ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file)
            series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file, ids[0])
            series_reader = sitk.ImageSeriesReader()
            series_reader.SetFileNames(series_file_names)
            series_reader.MetaDataDictionaryArrayUpdateOn()
            series_reader.LoadPrivateTagsOn()
            dicom_image = series_reader.Execute()
            space = dicom_image.GetSpacing()
            
            if pat_id in thickness_dict:
                if space[2]<thickness_dict[pat_id]:
                    thickness_dict[pat_id]=space[2]
                    thinner = True
            else:
                thickness_dict[pat_id]=space[2]
                thinner = True
            if thinner:
                print("Starting registration!",file)
    #        reader = sitk.ImageSeriesReader()
    #        dicom_names = reader.GetGDCMSeriesFileNames(file)
    #        reader.SetFileNames(dicom_names)
    #        reader.MetaDataDictionaryArrayUpdateOn()
    #        reader.LoadPrivateTagsOn()
    #        dicom_image = reader.Execute()
    #        print("Image read!")
            
                brain = segment_brain(1,dicom_image,model)
                
                brain = sitk.GetArrayFromImage(brain)
                brain = brain>0
                for i in range(0,brain.shape[0]):
                    brain[i,:,:] = binary_fill_holes(brain[i,:,:])
                    
                brain=np.array(brain,dtype='int16')
                brain = cv2.erode(brain,np.ones((3,3),np.uint8),iterations = 1)    
                
                ct_scan = sitk.GetArrayFromImage(dicom_image)
                
                slices=np.sum(brain,axis=1)
                slices=np.sum(slices,axis=1)
                
                slices = slices>0
                
                ct_scan = ct_scan
                ct_scan[brain==0] = 0   
                ct_scan = ct_scan[slices[:],:,:]
        
                 
                ct_scan = sitk.GetImageFromArray(ct_scan)
                
                ct_scan.SetSpacing(dicom_image.GetSpacing())

                ct_scan.SetOrigin(dicom_image.GetOrigin())
                
#                sitk.WriteImage(ct_scan, "skull_stripped_scan.mhd")
                
                #ct_scan.CopyInformation(dicom_image)
                #sitk.WriteImage(ct_scan,r"E:\MrClean_part1\R0003\1\test.mha")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                sitk.WriteImage(ct_scan,save_path+"\\skull_stripped_scan.mhd")
                
                moving_image_path = save_path+"\\skull_stripped_scan.mhd"
                end = time.time()
                print("Time to strip skull and save:",np.round(end - start,2))
                command = (elastix_location + " -f " + fixed_image_path + " -m " + moving_image_path + " -out " + save_path + " -p " + param_rigid + " -p " + param_affine)    
                print(subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read())
            #    start = time.time()    
            #    cmd = ("elastix.exe" + " -f " + fixed_image_path + " -m " + atlas_image_path + " -out " + save_path + " -p " + "Parameters_Rigid.txt" + " -p " + "Parameters_Affine.txt")
            #    cmd = ("elastix.exe")
            #    call(cmd, shell=True)
            #    end = time.time()
            #    print("Time to register and save:",end - start)
    except:
        print("Error!",file)
        error.append(file)
        np.save(error_name,error)