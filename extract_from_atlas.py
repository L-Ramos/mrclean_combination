# -*- coding: utf-8 -*-
"""
This code is for reading the registired images and extract the atlas locations

@author: laramos
"""

import SimpleITK as sitk
import numpy as np
import glob as glob
import os

cort_reg = sitk.ReadImage(r"\\amc.intra\users\L\laramos\home\Desktop\Atlas\Marielle_Ernst\HOcort.nii")
cort_reg = sitk.GetArrayFromImage(cort_reg)
print("Cort atlas has %d regions."%(len(np.unique(cort_reg))))

sub_reg = sitk.ReadImage(r"\\amc.intra\users\L\laramos\home\Desktop\Atlas\Marielle_Ernst\HOsub.nii")
sub_reg = sitk.GetArrayFromImage(sub_reg)
print("Sub atlas has %d regions."%(len(np.unique(sub_reg))))

other_reg = sitk.ReadImage(r"\\amc.intra\users\L\laramos\home\Desktop\Atlas\Marielle_Ernst\jhu_Antonia_orient.nii")
other_reg = sitk.GetArrayFromImage(other_reg)
print("Other atlas has %d regions."%(len(np.unique(other_reg))))

general_path = glob.glob(r'E:\MrClean_part1\R*')

for path in general_path:
    other_paths = os.listdir(path)    
    for images in other_paths:
         orig_scan = sitk.ReadImage(os.path.join(path,images,"result.1.mhd"))



orig_scan = sitk.ReadImage(r"E:\MrClean_part1\R0121\0\result.1.mhd")
scan = sitk.GetArrayFromImage(orig_scan)

scan = scan[sub_reg==13]
#scan[sub_reg!=13]=0

print(np.mean(scan[sub_reg==0]))
print(np.mean(scan[sub_reg==1]))
print(np.mean(scan[sub_reg==2]))
print(np.mean(scan[sub_reg==3]))
print(np.mean(scan[sub_reg==4]))

scan = sitk.GetImageFromArray(scan)

scan.SetSpacing(orig_scan.GetSpacing())
scan.SetOrigin(orig_scan.GetOrigin())

sitk.WriteImage(scan,r"E:\MrClean_part1\R0121\0\test.mhd")

