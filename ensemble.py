# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:30:53 2020

@author: laramos
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import scipy as sp

def Mean_Confidence_Interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.nanmean(a), sp.stats.sem(a,nan_policy = 'omit')
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

img = np.load(r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\RF_comb_imp_review_-01-RF_opt-roc_auc_und-W\image\probabilities_LR_0.npy")
clin = np.load(r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\RF_comb_imp_review_-01-RF_opt-roc_auc_und-W\clinical\probabilities_LR_0.npy")


        
#tn_img, fp_img, fn_img, tp_img = confusion_matrix(img[:,1], img[:,0]>=0.5).ravel() 
#print(tn_img,fp_img,fn_img,tp_img)
#
#auc_img = roc_auc_score(img[:,1], img[:,0])
#
#auc_clin = roc_auc_score(clin[:,1], clin[:,0])
#
#both = (clin[:,0]+img[:,0])/2
#
#auc_both = roc_auc_score(clin[:,1], both)
#
#tn_clin, fp_clin, fn_clin, tp_clin = confusion_matrix(clin[:,1], clin[:,0]>=0.5).ravel() 
#print(tn_clin,fp_clin,fn_clin,tp_clin)

correct = list()
wrong = list()
auc_comb = list()
auc_clin = list()
for j in range(0,10):
    img = np.load(r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\RF_comb_imp_review_-01-RF_opt-roc_auc_und-W\image\probabilities_RFC_"+str(j)+".npy")
    clin = np.load(r"\\amc.intra\users\L\laramos\home\Desktop\Combination_MrCLean\RF_comb_imp_review_-01-RF_opt-roc_auc_und-W\clinical\probabilities_RFC_"+str(j)+".npy")

    auc_clin.append(roc_auc_score(clin[:,1], clin[:,0]))
    
    tn_clin, fp_clin, fn_clin, tp_clin = confusion_matrix(clin[:,1], clin[:,0]>=0.5).ravel() 
    
    comb = np.zeros(clin.shape[0])
    
    for i in range(0,clin.shape[0]):
        diff_clin = abs(clin[i,0]-0.5)
        diff_img = abs(img[i,0]-0.5)
        if diff_img>diff_clin:
            comb[i]=img[i,0]
        else:
            comb[i]=clin[i,0]
    
#    comb = (clin[:,0]+img[:,0])/2        
    auc_comb.append(roc_auc_score(clin[:,1], comb))
        
    tn_both, fp_both, fn_both, tp_both = (confusion_matrix(clin[:,1], comb[:]>=0.5).ravel())
    
    correct.append((tp_both+tn_both) - (tp_clin+tn_clin))
    
    wrong.append((fp_clin+fn_clin) - (fp_both+fn_both))
    
    print("Correct clinical %d, Incorrect clinical %d"%(tp_clin+tn_clin,fp_clin+fn_clin))
    
    print("Correct  Both %d, Incorrect Both %d"%(tp_both+tn_both,fp_both+fn_both))

print(sum(correct))

print("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(auc_clin)))
print("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(auc_comb)))