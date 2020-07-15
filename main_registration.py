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

error_name = "E:\MrClean_part2\errors_final_3.npy"
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
            file = r"L:\basic\divi\Projects\mrclean\MRCLEAN-Registry\Imaging_Part1_Sorted\CTA_BL\R0021\26666532\1.3.6.1.4.1.40744.9.286343667699698292825853769978846503400"
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

#%% This part is for doing things by hand, for stuff that fails or look weird

import matplotlib.pyplot as plt

file_list = [r'L:\basic\divi\Projects\mrclean\MRCLEAN-Registry\Imaging_Part2_Sorted\CTA_BL\A_ASSESSEDBYCORELAB\R3121\6_other\DICOM']


for file in file_list: 
      
    pat_id = re.search('R[0-9][0-9][0-9][0-9]',file)[0]
    
    save_path = path_results+'\\'+pat_id
    
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file, ids[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    dicom_image = series_reader.Execute()
    space = dicom_image.GetSpacing()
    
    #dicom_image.SetSpacing([0.4609, 0.4609, 1])
    
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
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sitk.WriteImage(ct_scan,save_path+"\\skull_stripped_scan.mhd")
    
    
    #np.save("E:\MrClean_part1\errors_500-800.npy",error)
    #command = "elastix.exe -f E:\MrClean_part1\skull_stripped_atlas.mhd -m  E:\MrClean_part1\R0001\skull_stripped_scan.mhd -out E:\MrClean_part1\R0001 -p Parameters_Rigid_Manon.txt -p Parameters_Affine.txt"
    
    moving_image_path = save_path+"\\skull_stripped_scan.mhd"
    command = (elastix_location + " -f " + fixed_image_path + " -m " + moving_image_path + " -out " + save_path + " -p " + param_rigid + " -p " + param_affine)    
    print(subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read())



#%%



#this is for when the getspacing returns wrong values, so we fix by hand, sometimes it solves the error
ct_scan.SetSpacing((0.408203125, 0.408203125, 0.3333333))

ct_scan.SetOrigin(dicom_image.GetOrigin())

if not os.path.exists(save_path):
    os.makedirs(save_path)
sitk.WriteImage(ct_scan,save_path+"\\skull_stripped_scan.mhd")
#%%
#import pydicom as dicom
#from pydicom.filereader import read_dicomdir
#filepath =r""
#
##dicom_dir = read_dicomdir(filepath)
#
#dirname = filepath
#files = os.listdir(dirname)
#ds_list = [dicom.read_file(os.path.join(dirname, filename)) for filename in files]
#
#img_arr = np.zeros((182,512,512))
#
#for i,f in enumerate(ds_list):
#    img_arr[i,:,:]=ds_list[i].pixel_array
#    
#img = sitk.GetImageFromArray(img_arr)
#
#sitk.WriteImage(img,"skull_stripped_scan.mha")
#
#ds_list.save_as("skull_stripped_scan.mha")

#%% This is for the files that had an extra image

general_path = glob.glob(r'E:\Fixing_auto_part_1\R**',recursive=True)
general_path = general_path[0:len(general_path)-1]
path_results = r'E:\Fixing_auto_part_1\results'

#for file in general_path:
for key in path_dict:
        try:
            pat_id = re.search('R[0-9][0-9][0-9][0-9]',file)
            save_path = path_results+'\\'+pat_id[0]
            
            ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file)
            series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file, ids[0])
            series_reader = sitk.ImageSeriesReader()
            series_reader.SetFileNames(series_file_names)
            series_reader.MetaDataDictionaryArrayUpdateOn()
            series_reader.LoadPrivateTagsOn()
            dicom_image = series_reader.Execute()
            space = dicom_image.GetSpacing()
        
            brain = segment_brain(1,dicom_image,model)
                        
            brain = sitk.GetArrayFromImage(brain)
            brain = brain>0
            for i in range(0,brain.shape[0]):
                brain[i,:,:] = binary_fill_holes(brain[i,:,:])
                
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
            
            #ct_scan.CopyInformation(dicom_image)
            #sitk.WriteImage(ct_scan,r"E:\MrClean_part1\R0003\1\test.mha")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            sitk.WriteImage(ct_scan,save_path+"\\skull_stripped_scan.mhd")
            
            moving_image_path = save_path+"\\skull_stripped_scan.mhd"
            command = (elastix_location + " -f " + fixed_image_path + " -m " + moving_image_path + " -out " + save_path + " -p " + param_rigid + " -p " + param_affine)    
            print(subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read())
        except:
            print("Failed:",key)


#%% This part is to fix the image that have the topogram
    
import shutil,os

def copytree2(source,dest):
    os.mkdir(dest)
    dest_dir = os.path.join(dest,os.path.basename(source))
    shutil.copytree(source,dest_dir)
    
import pandas as pd
fixing_path = r'E:\Fixing_auto_part_2'
path_results = r'E:\Fixing_auto_part_2\results'

failed = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\imaging\failed registration.csv",header=None)
failed = list(failed[0])
inside = False
wrong_path = list()
wrong_dim=list()
wrong = list()
for key in path_dict:
    try:
        file = path_dict[key]
        pat_id = re.search('R[0-9][0-9][0-9][0-9]',file[int(len(file)/2)])
        save_path = path_results+'\\'+pat_id[0]
        fix_path = fixing_path+'\\'+pat_id[0]
        if pat_id[0] not in failed:
            #Compares the beginningm middel and end of the dicom thickness, these are topograms
            f = sorted(glob.glob(file[int(len(file)/2)]+'\\*.dcm'))
            img_ref_beg = sitk.ReadImage(f[0])
            img_ref_mid = sitk.ReadImage(f[int(len(f)/2)])
            img_ref_end = sitk.ReadImage(f[len(f)-1])
            inside=True
            if img_ref_mid.GetSpacing() != img_ref_beg.GetSpacing() :
                #if it is a topogram, copy to my folder, delete topogram and runs the registration
                #os.makedirs(fix_path)
                copytree2(file[int(len(file)/2)],fix_path)
                _,folder = os.path.split(file[int(len(file)/2)]) 
                def_f = sorted(glob.glob(fix_path+'\\*\\*.dcm',recursive=True))
                os.unlink(def_f[0])
                inside=True
            else:
                if img_ref_mid.GetSpacing() != img_ref_end.GetSpacing():
                    #os.makedirs(fix_path)
                    copytree2(file[int(len(file)/2)],fix_path)
                    def_f = sorted(glob.glob(fix_path+'\\*.dcm'))
                    os.unlink(def_f[len(f)-1])
                    inside=True
        if inside:
                print(key)
                ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(fix_path+'\\'+folder)       
                series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(fix_path+'\\'+folder, ids[0])
                series_reader = sitk.ImageSeriesReader()
                series_reader.SetFileNames(series_file_names)
                series_reader.MetaDataDictionaryArrayUpdateOn()
                series_reader.LoadPrivateTagsOn()
                dicom_image = series_reader.Execute()
                space = dicom_image.GetSpacing()
            
                brain = segment_brain(1,dicom_image,model)
                           
                brain = sitk.GetArrayFromImage(brain)
                brain = brain>0
                for i in range(0,brain.shape[0]):
                    brain[i,:,:] = binary_fill_holes(brain[i,:,:])
                    
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
                
                #ct_scan.CopyInformation(dicom_image)
                #sitk.WriteImage(ct_scan,r"E:\MrClean_part1\R0003\1\test.mha")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                sitk.WriteImage(ct_scan,save_path+"\\skull_stripped_scan.mhd")
                
                moving_image_path = save_path+"\\skull_stripped_scan.mhd"
                command = (elastix_location + " -f " + fixed_image_path + " -m " + moving_image_path + " -out " + save_path + " -p " + param_rigid + " -p " + param_affine)    
                print(subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read())      
                inside=False
    except:
        print("Failed:",key)
        wrong.append(key)
 

#ids_error = list()
#
#for e in error:
#    p_id = re.search('R[0-9][0-9][0-9][0-9]',e)
#    ids_error.append(p_id[0])
#    
#%%MISSING REGISTRATION PART 2
import glob
import os        
import re
import pickle

path_mr = r'L:\basic\divi\Projects\mrclean\MRCLEAN-Registry\Imaging_Part2_Sorted\CTA_BL\A_ASSESSEDBYCORELAB'

id_list = list()
id_done_list = list()

already_done = sorted(os.listdir(r'E:\MrClean_part2'))
for f in already_done:
    id_p = re.search('R[0-9][0-9][0-9][0-9]',f)
    if id_p:
        id_done_list.append(id_p[0])
    
path_list = sorted(os.listdir(path_mr))
for f in path_list:
    id_list.append(re.search('R[0-9][0-9][0-9][0-9]',f)[0])

from collections import defaultdict
path_dict = dict()   

s_id_done = set(id_done_list)
s_id = set(id_list) 
to_do_list = s_id - s_id_done 
to_do_list = sorted(list(to_do_list))

to_do_list = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\failed_part2.csv",header=None)
to_do_list = sorted(list(to_do_list[0]))

for i,p_id in enumerate(to_do_list):
        print(i,"   ",len(to_do_list))
        list_paths = list()
        complete_path = path_mr+'\\'+p_id
        all_f = glob.glob(complete_path+'\\**\\*.dcm',recursive = True)
        for f in all_f:
            h,t=os.path.split(f)
            if h not in list_paths:
                list_paths.append(h)
        path_dict[p_id] = list_paths

#note for myself: path_dict has the missing images that need to fo through the registration pipeline                       
outfile = open(r"E:\MrClean_part2\path_files_complete.npy",'wb')
pickle.dump(path_dict,outfile)
#%%


#path_dict = np.load(r"E:\MrClean_part2\path_files_2_round.npy",allow_pickle=True) 
        
#clean_dict = dict()
#
#for key in path_dict:
#    if key not in list(failed[0]):
#        clean_dict[key] = path_dict[key]
#        
    
#%%   plotting and saving for check up

import glob
import SimpleITK as sitk
import numpy as np
import pandas as pd
import re
from PIL import Image 
import matplotlib.pyplot as plt


image_list  = glob.glob(r"E:\MrClean_part2\*\result.1.mhd")

for k,f in enumerate(image_list):  
    p_id = re.search('R[0-9][0-9][0-9][0-9]',f)
    img = sitk.ReadImage(f)  
    img_arr = sitk.GetArrayFromImage(img)
    #save_img  = Image.fromarray(img_arr[100,:,:],'RGB')
    #save_img.save('my.png')
    plt.imshow(img_arr[80,:,:],'bone')
    plt.savefig(r'E:\MrClean_part2\Check\\'+p_id[0]+'.png')
        