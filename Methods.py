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
#import eli5   

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
        'n_estimators': ([200,400,500,600,800,1000,1200,1400]),
        'max_features': (['auto', 'sqrt', 'log2']),                   # precomputed,'poly', 'sigmoid'
        'max_depth':    ([10,20,30,40, 50, 60, 70, 80, 90, 100, None]),
        'criterion':    (['gini', 'entropy']),
        'min_samples_split':  [2,4,6,8],
        'min_samples_leaf':   [2,4,6,8,10]}
        return(tuned_parameters)
        
def get_LR(): 
       
        tuned_parameters = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        return(tuned_parameters)
        
def get_NN():
    
        tuned_parameters = {
        'activation': (['relu','logistic']),
        'hidden_layer_sizes':([[60,120,60],[60,120,120],[60,60],[60,120]]),
        #'hidden_layer_sizes':([[131,191,131],[131,231,131],[131,131,131]]),
        'alpha':     ([0.01, 0.001, 0.0001]),
        'batch_size':         [32,64],
        'learning_rate_init':    [0.01, 0.001],
        'solver': ["adam"]}
        return(tuned_parameters)
        
#def get_LASSO():
    
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
        #clf_grid =  RandomizedSearchCV(clf_g, tuned_parameters, cv=splits,random_state=random_state,
        #                   scoring='%s' % opt[0],n_jobs=n_jobs)        
        clf_grid =  RandomizedSearchCV(clf_g, tuned_parameters, cv=splits,random_state=random_state,
                           scoring='%s' % opt[0],n_jobs=n_jobs,n_iter=50,verbose=1)
                                          
        clf_grid.fit(x_train, y_train)
        #print("Score",clf.best_score_)
        end_rfc = time.time()
        
        print("Time to process: ",end_rfc - start_rfc)
        
        with open(path_results+"parameters_"+name+".txt", "a") as file:
            for item in clf_grid.best_params_:
              file.write(" %s %s " %(item,clf_grid.best_params_[item] ))
            file.write("\n")
            
        #clf = clf_g(**clf_grid.best_params_,random_state=random_state)
        clf = clf_grid.best_estimator_
        
                    
        #clf_t = clf_g(**clf_grid.best_params_,random_state=random_state)
        clf = clf.fit(x_train,y_train)

                             
        if name=="SVM":
            decisions = clf.decision_function(x_test)
            probas=\
            (decisions-decisions.min())/(decisions.max()-decisions.min())
        else:
             probas = clf.predict_proba(x_test)[:, 1]


                
        self.model = clf        
        preds = clf.predict(x_test)        
        m.clf_f1_score[itera]=f1_score(y_test, preds)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()        
        m.clf_sens[itera]=tp/(tp+fn)
        m.clf_spec[itera]=tn/(tn+fp)
        m.clf_ppv[itera]=tp/(tp+fp)
        m.clf_npv[itera]=tn/(tn+fn)                        
        m.clf_auc[itera] = roc_auc_score(y_test,probas)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, probas)  
        
        m.clf_brier[itera] = brier_score_loss(y_test, probas)   
                
        save_prob = np.concatenate((probas.reshape(-1,1),y_test.reshape(-1,1)),axis = 1)
        
        
      
        #np.save(path_results+"probabilities_"+name+"_"+str(itera)+".npy",probas)
        
        np.save(path_results+"probabilities_"+name+"_"+str(itera)+".npy",save_prob)
        #np.save(path_results+"probabilities_train"+name+"_"+str(itera)+".npy",save_prob_train)

        #np.save(path_results+"feature_importance"+name+"_"+str(itera)+str(i)+".npy",clf.coef_)
        #joblib.dump(clf,path_results+'clf_'+name+str(itera)+str(i))
        return(fpr_rf,tpr_rf,probas,clf)
        
        
        
    def __init__(self,run,name ,x_train,y_train,x_test,y_test,itera,cv,mean_tprr,m,path_results,opt,und,final_cols):
        if run:
            opt=[opt]            
            if name == 'RFC':
                print("RFC Grid Search")                
                tuned_parameters = get_RFC()
                if und=='W':
                    clf = RandomForestClassifier(n_estimators=25, oob_score = True,random_state=random_state,class_weight='balanced')   
                else:
                    clf = RandomForestClassifier(n_estimators=25, oob_score = True,random_state=random_state)   
            else:
                if name == 'SVM':
                    print("SVM Grid Search")
                    tuned_parameters = get_SVC()
                    if und=='W':
                        clf = SVC(random_state=random_state,class_weight='balanced')
                    else:                    
                        clf = SVC(random_state=random_state)
                else:
                    if name == 'LR':
                        print("LR Grid Search")
                        tuned_parameters = get_LR()
                        if und=='W':
                            clf = LogisticRegression(random_state=random_state,max_iter=5000,class_weight='balanced') 
                        else:
                            clf = LogisticRegression(random_state=random_state,max_iter=5000) 
                    else:
                        if name == 'NN':
                            print("NN Grid Search")
                            tuned_parameters = get_NN()                                                        
                            clf = MLPClassifier(hidden_layer_sizes=(x_train.shape[1]),max_iter=5000,batch_size=32,random_state=random_state )                         
                                    
                        else:                    
                            if name == 'XGB':
                                print("XGB Grid Search")
                                tuned_parameters = get_XGB()
                                if und=='W':
                                    clf = xgb.XGBClassifier(random_state=random_state,scale_pos_weight=(y_train.shape[0]-sum(y_train))/sum(y_train)) 
                                else:                                    
                                    clf = xgb.XGBClassifier(random_state=random_state) 
                        
            self.name=name
            m.run=True
            fpr_rf,tpr_rf,probas_t,clf=self.RandomGridSearch(x_train,y_train,x_test,y_test,cv,path_results,m,itera,clf,name,tuned_parameters,opt,final_cols)
            print("Done Grid Search")
            print("Done testing - "+ name, m.clf_auc[itera])
            mean_fpr = np.linspace(0, 1, 100) 
            m.mean_tpr += interp(mean_fpr, fpr_rf,tpr_rf)
            m.mean_tpr[0] = 0.0
        else:
            self.name='NONE'
            self.clf=0



def Mean_Confidence_Interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.nanmean(a), sp.stats.sem(a,nan_policy = 'omit')
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def Print_Results_Excel(m,splits,names,path_results,l):    
    colors=['darkorange','blue','green','black','yellow']
    book = xlwt.Workbook(encoding="utf-8")    
    sheet1 = book.add_sheet("Sheet 1")
    #path_results_txt=path_results+path_results[2:len(path_results)-2]+str(l)+".xls"
    path_results_txt = os.path.join(path_results,('results'+str(l)+".xls"))
    sheet1.write(0, 0, "Methods")
    sheet1.write(0, 1, "AUC 95% CI ")
    sheet1.write(0, 2, "Brier ")
    sheet1.write(0, 3, "F1-Score")
    sheet1.write(0, 4, "Sensitivity")
    sheet1.write(0, 5, "Specificity")
    sheet1.write(0, 6, "PPV")
    sheet1.write(0, 7, "NPV")
    #sheet1.write(0, 8, "F1-Score_f1")
    #sheet1.write(0, 9, "Sensitivity_f1")
    #sheet1.write(0, 10, "Specificity_f1")
    #sheet1.write(0, 11, "PPV_f1")
    #sheet1.write(0, 12, "NPV_f1")
    #sheet1.write(0, 13, "F1-Score_spec")
    #sheet1.write(0, 14, "Sensitivity_spec")
    #sheet1.write(0, 15, "Specificity_spec")
    #sheet1.write(0, 16, "PPV_spec")
    #sheet1.write(0, 17, "NPV_spec")
    #Spec and sensitivty are inverted because of the label
    for i in range(0,len(names)):        
        print(i,names[i])
        sheet1.write(i+1,0,(names[i])) 
        sheet1.write(i+1,1,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_auc.reshape(-1)))))              
        sheet1.write(i+1,2,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_brier.reshape(-1)))))              
        sheet1.write(i+1,3,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_f1_score.reshape(-1)))))              
        sheet1.write(i+1,4,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_sens.reshape(-1)))))              
        sheet1.write(i+1,5,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_spec.reshape(-1)))))              
        sheet1.write(i+1,6,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_ppv.reshape(-1)))))              
        sheet1.write(i+1,7,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(m[i].clf_npv.reshape(-1)))))              
        
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

        np.save(file=os.path.join(path_results,('AUCs_'+names[i]+'.npy')),arr=m[i].clf_auc)
        #np.save(file=path_results+'Thresholds_'+names[i]+'.npy',arr=m[i].clf_thresholds)
        mean_tpr=m[i].mean_tpr
        mean_tpr /= splits
        mean_tpr[-1] = 1.0
        #frac_pos_rfc  /= skf.get_n_splits(X, Y)
        mean_fpr = np.linspace(0, 1, 100) 
        mean_auc_rfc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color=colors[i],lw=2, label=names[i]+' (area = %0.2f)' % mean_auc_rfc)
        plt.legend(loc="lower right")
        np.save(file=os.path.join(path_results,('tpr_'+names[i]+'.npy')),arr=mean_tpr)
        np.save(file=os.path.join(path_results,('fpr_'+names[i]+'.npy')),arr=mean_fpr)
        if names[i]=='RFC':
            np.save(file=os.path.join(path_results,('Feat_Importance'+names[i]+'.npy')),arr=m[i].feat_imp)
    book.save(path_results_txt)        
    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
    #plt.show() 

