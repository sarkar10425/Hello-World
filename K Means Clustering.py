#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns



# In[3]:


from sklearn.datasets import make_blobs


# In[4]:


data=make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.8,random_state=101)


# In[5]:


data


# In[7]:


data[0].shape


# In[10]:


plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')


# In[9]:


data[1]


# In[11]:


from sklearn.cluster import KMeans


# In[35]:


# Change the n_clusters to 1,2,3,4,5,6.7,8.......
kmeans=KMeans(n_clusters=4)


# In[36]:


kmeans.fit(data[0])


# In[37]:


kmeans.cluster_centers_


# In[38]:


kmeans.labels_


# In[39]:


fig,(axis1,axis2)=plt.subplots(1,2,sharey=True,figsize=(10,6))
axis1.set_title('K Means')
axis1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')

axis2.set_title('Original')
axis2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')




# In[ ]:





# In[ ]:




