# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:43:17 2019

@author: laramos
"""


#TSNE First experiment, tsne visualization, doesnt work well =X

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
target = data.target


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_pickle(r"\\amc.intra\users\L\laramos\home\Desktop\New_data_Mrclean\\imaging\features_HOcort.pkl")

df=df.drop(['ID'],axis=1)

X_embedded = TSNE(n_components=2,perplexity=30, n_iter=5000).fit_transform(df)


plt.scatter(X_embedded[:,0],X_embedded[:,1],  c = target, 
            cmap = "coolwarm", edgecolor = "None", alpha=0.35)
plt.colorbar()
plt.title('TSNE Scatter Plot')
plt.show()
