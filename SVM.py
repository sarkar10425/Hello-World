#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns



# In[3]:


cancer=pd.read_csv(r'D:\Datasets\tumour.csv')


# In[4]:


cancer.head()


# In[5]:


cancer.info()


# In[7]:


from sklearn.datasets import load_breast_cancer


# In[9]:


cancer=load_breast_cancer()


# In[10]:


cancer.keys()


# In[11]:


print(cancer['DESCR'])


# In[12]:


cancer['target']


# In[13]:


cancer['target_names']


# In[16]:


cancer['data']


# In[15]:


df_feat=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[17]:


df_feat.head()


# In[18]:


from sklearn.model_selection import train_test_split


# In[21]:


X=df_feat
y=cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[22]:


from sklearn.svm import SVC


# In[23]:


model=SVC()


# In[24]:


model.fit(X_train,y_train)


# In[26]:


y_pred=model.predict(X_test)


# In[27]:


from sklearn.metrics import classification_report,confusion_matrix


# In[28]:


print(classification_report(y_pred,y_test))
print('\n')
print(confusion_matrix(y_pred,y_test))


# In[29]:


from sklearn.metrics import accuracy_score


# In[30]:


acc=accuracy_score(y_pred,y_test)


# In[31]:


acc*100


# In[32]:


from sklearn.grid_search import GridSearchCV


# In[33]:


pram_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}


# In[34]:


grid=GridSearchCV(SVC(),pram_grid,verbose=3)


# In[35]:


grid.fit(X_train,y_train)


# In[36]:


grid.best_params_


# In[37]:


grid.best_estimator_


# In[38]:


grid_predictions=grid.predict(X_test)


# In[39]:


print(classification_report(grid_predictions,y_test))
print('\n')
print(confusion_matrix(grid_predictions,y_test))


# In[40]:


accuracy=accuracy_score(y_test,grid_predictions)


# In[41]:


accuracy*100


# In[ ]:




