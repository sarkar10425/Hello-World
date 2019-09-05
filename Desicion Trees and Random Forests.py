#!/usr/bin/env python
# coding: utf-8

# # Project 1

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns



# In[3]:


kyp=pd.read_csv(r'D:\Datasets\kyphosis.csv')


# In[4]:


kyp.head()


# In[5]:


kyp.info()


# In[6]:


sns.pairplot(data=kyp,hue='Kyphosis')


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X=kyp.drop('Kyphosis',axis=1)


# In[9]:


y=kyp['Kyphosis']


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[11]:


from sklearn.tree import DecisionTreeClassifier


# In[12]:


dtree=DecisionTreeClassifier()


# In[13]:


dtree.fit(X_train,y_train)


# In[14]:


y_pred=dtree.predict(X_test)


# In[15]:


from sklearn.metrics import accuracy_score


# In[16]:


acc=accuracy_score(y_test,y_pred)


# In[17]:


acc*100


# In[18]:


from sklearn.metrics import classification_report,confusion_matrix


# In[19]:


print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


# In[20]:


from sklearn.ensemble import RandomForestClassifier


# In[21]:


rfc=RandomForestClassifier(n_estimators=200)


# In[22]:


rfc.fit(X_train,y_train)


# In[23]:


y_pred=rfc.predict(X_test)


# In[24]:


acc=accuracy_score(y_test,y_pred)


# In[25]:


acc*100


# In[26]:


print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


# In[27]:


error=[]
for i in range(1,50):
    rfc=RandomForestClassifier(n_estimators=i)
    rfc.fit(X_train,y_train)
    y_pred=rfc.predict(X_test)
    
    error.append(np.mean(y_test!=y_pred))


# In[28]:


plt.figure(figsize=(10,9))
plt.plot(range(1,50),error,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12,markerfacecolor='red')


# In[29]:


rfc=RandomForestClassifier(n_estimators=9)
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
acc=accuracy_score(y_test,y_pred)
acc*100


# #  Project 2 

# In[33]:


loan=pd.read_csv(r'D:\Datasets\loan.csv')


# In[34]:


loan


# In[35]:


loan.info()


# In[ ]:




