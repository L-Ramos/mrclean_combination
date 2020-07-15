"""
This is a new version of methods using one single function
"""

#from tensorflow import keras

#import preprocess_input

import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV                               
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import brier_score_loss,f1_score
import random as rand
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import warnings
import time
from scipy.stats import randint as sp_randint
import scipy as sp
warnings.filterwarnings("ignore", category=DeprecationWarning)
from scipy import interp 
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFECV
from sklearn.naive_bayes import GaussianNB
import xlwt
from sklearn.metrics import make_scorer
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
import eli5
import os

from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error


# =============================================================================
# 
# def performance_metric(y_true, y_predict):
#     """ Calculates and returns the performance score between 
#         true and predicted values based on the metric chosen. 
#     """
#     
#     
#     tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
# 
#     #sens=mat[0,0]/(mat[0,0]+mat[1,0])
# 
#     spec=tn/(tn+fp)
#     
#     # Return the score
#     return(spec)
# =============================================================================
    
#Public variables    
#scoring_fnc = make_scorer(performance_metric)

#score = scoring_fnc

n_jobs=-2

random_state=1


def get_SVC():
    
        tuned_parameters = {
        'C':            ([0.1, 0.01, 0.001, 1, 10, 100]),
        'kernel':       ['linear', 'rbf','poly'],                
        'degree':       ([1,2,3,4,5,6]),
        'gamma':         [1, 0.1, 0.01, 0.001, 0.0001]
        #'tol':         [1, 0.1, 0.01, 0.001, 0.0001],
        }
        return(tuned_parameters)
    
def get_RFC():
    
        tuned_parameters = {
        'n_estimators': ([100,200,300]),
        'max_features': (['auto', 'sqrt', 'log2']),                   # precomputed,'poly', 'sigmoid'
        'max_depth':    ([10,20,30]),
        #'criterion':    (['mse', 'mae']),
        'min_samples_split':  [2,4,6],
        'min_samples_leaf':   [2,4,6]}
        return(tuned_parameters)
        
def get_Lasso(): 
       
        tuned_parameters = {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
        return(tuned_parameters)
        
def get_NN():
    
        tuned_parameters = {
        'activation': (['relu','logistic','tanh']),
        'hidden_layer_sizes':([[90,180,90],[90,120,90],[90,90],[90,180]]),
        #'hidden_layer_sizes':([[131,191,131],[131,231,131],[131,131,131]]),
        'alpha':     ([0.01, 0.001, 0.0001]),
        'batch_size':         [32,64],
        'learning_rate_init':    [0.01, 0.001],
        'solver': ["adam"]}
        return(tuned_parameters)
        
    
def get_XGB():
    
        tuned_parameters = {
        'learning_rate': ([0.1, 0.01, 0.001]),
        #'gamma': ([100,10,1, 0.1, 0.01, 0.001]),                  
        #'max_depth':    ([3,5,10,15]),
        #'subsample ':    ([0.5,1]),
        #'reg_lambda ':  [1,10,100],
        #'alpha ':   [1,10,100],
        
        'min_child_weight': [1, 5, 10],
        'gamma': [0, 0.5, 1, 1.5, 2, 5],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.3,0.4,0.5,0.6,0.7,0.8],
        'max_depth': [3, 5, 7, 9, 10]}
        return(tuned_parameters)

#def shape_visualization():
            
#        import shap
#        #testing feature importance with xgb
#        x_train_t = pd.DataFrame(x_train,columns=new_cols)
#        #f = plt.figure(figsize=(25, 19))
#        #xgboost.plot_importance(clf_t,importance_type="gain")
#       
#        explainer = shap.TreeExplainer(clf_t)
#        shap_values = explainer.shap_values(x_train_t)
#        shap.summary_plot(shap_values, x_train_t, plot_type="bar")
#        #end of test


#def random_features_permutation():
#    
#    model = clf                  
#    
#    f = pd.DataFrame()
#    f['name']=name
#    
#    from eli5.sklearn import PermutationImportance
#    perm = PermutationImportance(model, random_state=1,scoring="roc_auc").fit(x_train,y_train)
#    new_cols = final_cols
#    html = eli5.explain_weights(perm, feature_names = new_cols)  
#    for imp in range(len(html.feature_importances.importances)):
#        weights.append(html.feature_importances.importances[imp].weight)
#        stds.append(html.feature_importances.importances[imp].std)
#        names.append(html.feature_importances.importances[imp].feature)
#    
#    import_frame = pd.DataFrame(list(zip(names,weights,stds)))
#    import_frame.columns = ['name','weight','std']
#    import_frame.to_excel(path_results+'features_'+name+'_'+str(itera)+'.xls')
        
class Pipeline: 
 
    def RandomGridSearch(self,x_train,y_train,x_test,y_test,splits,path_results,m,itera,clf_g,name,tuned_parameters,opt,final_cols):
        """
        This function looks for the best set o parameters for RFC method
        Input: 
            X: training set
            Y: labels of training set
            splits: cross validation splits, used to make sure the parameters are stable
        Output:
            clf.best_params_: dictionary with the parameters, to use: param_svm['kernel']
        """    
        
        start_rfc = time.time()                  
       
        clf_grid = RandomizedSearchCV(clf_g, tuned_parameters, cv=splits,random_state=random_state,
                           scoring='%s' % 'neg_root_mean_squared_error',n_jobs=n_jobs,verbose=1,n_iter=10)
        
        clf_grid = RandomizedSearchCV(clf, tuned_parameters, cv=5,random_state=42,
                           scoring='%s' % 'explained_variance',n_jobs=-2,verbose=5,n_iter=10) 
        
        clf_grid.fit(X_train_imp, y_train)
                                          
        clf_grid.fit(x_train, y_train)
        print("Score",clf_grid.best_score_)
        end_rfc = time.time()
        
        print("Time to process: ",end_rfc - start_rfc)
        
        with open(os.path.join(path_results,("parameters_"+name+".txt")), "a") as file:
            for item in clf_grid.best_params_:
              file.write(" %s %s " %(item,clf_grid.best_params_[item] ))
            file.write("\n")
            
        clf = clf_grid.best_estimator_
                             

                
        self.model = clf        
        preds = clf.predict(x_test)        
       
        save_prob = np.concatenate((preds.reshape(-1,1),y_test.reshape(-1,1)),axis = 1)
        
        
        m.clf_r2[itera] = clf.score(x_test, y_test)
        m.clf_mae[itera] = mean_absolute_error(y_test,preds)
        m.clf_mse[itera] = mean_squared_error(y_test,preds)
        m.clf_mdae[itera] = median_absolute_error(y_test,preds)
        
        m.clf_r_mae[itera] = mean_absolute_error(y_test,np.round(preds))
        m.clf_r_mse[itera] = mean_squared_error(y_test,np.round(preds))
        m.clf_r_mdae[itera] = median_absolute_error(y_test,np.round(preds))

        #np.save(path_results+"probabilities_"+name+"_"+str(itera)+".npy",probas)
        
        np.save(os.path.join(path_results,("probabilities_"+name+"_"+str(itera)+".npy")),save_prob)
        
        #np.save(path_results+"probabilities_train"+name+"_"+str(itera)+".npy",save_prob_train)

        #np.save(path_results+"feature_importance"+name+"_"+str(itera)+str(i)+".npy",clf.coef_)
        #joblib.dump(clf,path_results+'clf_'+name+str(itera)+str(i))
        return(preds,clf)
        
        
        
    def __init__(self,run,name ,x_train,y_train,x_test,y_test,itera,cv,mean_tprr,m,path_results,opt,und,final_cols):
        if run:
            opt=[opt]            
            if name == 'RFC':
                print("RFC Grid Search")                
                tuned_parameters = get_RFC()
                clf = RandomForestRegressor(n_estimators=25, oob_score = True,random_state=random_state)   
   
            else:
                if name == 'SVM':
                    print("SVM Grid Search")
                    tuned_parameters = get_SVC()
                    clf = svm.SVR()

                else:
                    if name == 'LR':
                        print("LR Grid Search")
                        tuned_parameters = get_Lasso()
                        clf = linear_model.LassoLars() 
                    else:
                        if name == 'NN':
                            print("NN Grid Search")
                            tuned_parameters = get_NN()                                                        
                            clf = MLPRegressor(hidden_layer_sizes=(x_train.shape[1]),max_iter=2000,batch_size=32,random_state=random_state )                         
                                    
                        else:                    
                            if name == 'XGB':
                                print("XGB Grid Search")
                                tuned_parameters = get_XGB()
                                clf = xgb.XGBRFRegressor(random_state=random_state) 

            self.name=name
            m.run=True
            preds_t,clf=self.RandomGridSearch(x_train,y_train,x_test,y_test,cv,path_results,m,itera,clf,name,tuned_parameters,opt,final_cols)
            print("Done Grid Search")
            print("Done testing - "+ name, m.clf_r2[itera])
        else:
            self.name='NONE'
            self.clf=0



def Mean_Confidence_Interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.nanmean(a), sp.stats.sem(a,nan_policy = 'omit')
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def Print_Results_Excel(m,splits,names,path_results,l,data_use):    
    
    book = xlwt.Workbook(encoding="utf-8")    
    sheet1 = book.add_sheet("Sheet 1")
    #path_results_txt=path_results+path_results[2:len(path_results)-2]+str(l)+".xls"
    path_results_txt = os.path.join(path_results,('results'+str(l)+data_use+".xls"))
    sheet1.write(0, 0, "Methods")
    sheet1.write(0, 1, "R2")
    sheet1.write(0, 2, "MAE")
    sheet1.write(0, 3, "MSE")
    sheet1.write(0, 4, "MDAE")
    sheet1.write(0, 5, "Rounded MAE")
    sheet1.write(0, 6, "Rounded MSE")
    sheet1.write(0, 7, "Rounded MDAE")
    #Spec and sensitivty are inverted because of the label
    for i in range(0,len(names)):        
        print(i,names[i])
        sheet1.write(i+1,0,(names[i])) 
        sheet1.write(i+1,1,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_r2.reshape(-1)))))              
        sheet1.write(i+1,2,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_mae.reshape(-1)))))              
        sheet1.write(i+1,3,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_mse.reshape(-1)))))              
        sheet1.write(i+1,4,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_mdae.reshape(-1)))))
        sheet1.write(i+1,5,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_r_mae.reshape(-1)))))              
        sheet1.write(i+1,6,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_r_mse.reshape(-1)))))              
        sheet1.write(i+1,7,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_r_mdae.reshape(-1)))))                
        # sheet1.write(i+1,5,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_spec.reshape(-1)))))              
        # sheet1.write(i+1,6,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_ppv.reshape(-1)))))              
        # sheet1.write(i+1,7,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_npv.reshape(-1)))))              
        
#        sheet1.write(i+1,8,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].f1_score_f1.reshape(-1)))))              
#        sheet1.write(i+1,9,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].sens_f1.reshape(-1)))))              
#        sheet1.write(i+1,10,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].spec_f1.reshape(-1)))))              
#        sheet1.write(i+1,11,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_ppv_f1.reshape(-1)))))              
#        sheet1.write(i+1,12,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_npv_f1.reshape(-1)))))              
#        
#        sheet1.write(i+1,13,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].f1_score_spec.reshape(-1)))))              
#        sheet1.write(i+1,14,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].sens_spec.reshape(-1)))))              
#        sheet1.write(i+1,15,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].spec_spec.reshape(-1)))))
#        sheet1.write(i+1,16,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_ppv_spec.reshape(-1)))))              
#        sheet1.write(i+1,17,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_npv_spec.reshape(-1)))))                                    

        np.save(file=path_results+'AUCs_'+names[i]+'.npy',arr=m[i].clf_auc)
        #np.save(file=path_results+'Thresholds_'+names[i]+'.npy',arr=m[i].clf_thresholds)
              
#      if names[i]=='RFC':
#           np.save(file=path_results+'Feat_Importance'+names[i]+'.npy',arr=m[i].feat_imp)
    book.save(path_results_txt)        
    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
    #plt.show() 

