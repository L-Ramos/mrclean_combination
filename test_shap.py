# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:22:31 2020

@author: laramos
"""

from sklearn.linear_model import LogisticRegression
import shap
clf = LogisticRegression()
for c in cols_use:
    frame_train[c] = frame_train[c].astype('float32')
x_train = np.array(frame_train[cols_use])
y_train = np.array(frame_train[label_name],dtype='int16')
clf = clf.fit(x_train,y_train)

explainer = shap.LinearExplainer(clf, x_train)
shap_values = explainer.shap_values(x_train)
shap.summary_plot(shap_values, x_train)
clf.predict(X_test_imp)


explainer = shap.KernelExplainer(clf.predict, x_train)
shap_values = explainer.shap_values(x_train)
shap.summary_plot(shap_values, x_train)
clf.predict(X_test_imp)