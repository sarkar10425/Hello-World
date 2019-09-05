#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns



# In[5]:


df=pd.read_csv(r'D:\Datasets\data.csv',index_col=0)


# In[8]:


df.head(100)


# In[10]:


df1=pd.get_dummies(df['diagnosis'],drop_first=True)


# In[12]:


df1.head(50)


# In[13]:


df=pd.concat([df,df1],axis=1)


# In[15]:


df.drop('diagnosis',axis=1,inplace=True)


# In[17]:


df


# In[18]:


df.drop('Unnamed: 32',axis=1,inplace=True)


# In[19]:


df.head(2)


# In[20]:


from sklearn.preprocessing import StandardScaler


# In[21]:


scaler=StandardScaler()


# In[22]:


scaler.fit(df.drop('M',axis=1))


# In[24]:


scaled_features=scaler.transform(df.drop('M',axis=1))


# In[28]:


scaled_features


# In[30]:


df_feat=pd.DataFrame(scaled_features,columns=df.columns[:-1])


# In[32]:


df_feat.head()


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


X=df_feat
y=df['M']


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[38]:


from sklearn.neighbors import KNeighborsClassifier


# In[201]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[202]:


knn.fit(X_train,y_train)


# In[203]:


y_pred=knn.predict(X_test)


# In[204]:


y_pred


# In[205]:


from sklearn.metrics import classification_report,confusion_matrix


# In[206]:


print(classification_report(y_test,y_pred))


# In[207]:


print(confusion_matrix(y_test,y_pred))


# In[208]:


from sklearn.metrics import accuracy_score


# In[209]:


acc=accuracy_score(y_test,y_pred)


# In[210]:


acc*100


# In[ ]:


error=[]

for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    
    error.append(np.mean(y_pred!=y_test))


# In[145]:


error


# In[146]:


plt.figure(figsize=(10,7))
plt.plot(range(1,40),error,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12,markerfacecolor='red')
plt.xlabel('K-value')
plt.ylabel('Error_rate')
plt.title('Error_rate vs K_value curve')


# In[147]:


X=df.drop('M',axis=1)


# In[148]:


y=df['M']


# In[149]:


from sklearn.model_selection import train_test_split


# In[150]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[151]:


from sklearn.neighbors import KNeighborsClassifier


# In[184]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[185]:


knn.fit(X_train,y_train)


# In[186]:


y_pred=knn.predict(X_test)


# In[187]:


y_pred


# In[188]:


from sklearn.metrics import accuracy_score


# In[189]:


acc=accuracy_score(y_test,y_pred)


# In[190]:


acc*100


# In[159]:


test=y_test.as_matrix()


# In[160]:


test.shape


# In[161]:


y_pred.shape


# In[162]:


a=np.vstack((test,y_pred))
print(a)


# In[163]:


error=[]

for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    
    error.append(np.mean(y_pred!=y_test))
    


# In[164]:


error


# In[165]:


plt.figure(figsize=(10,7))
plt.plot(range(1,40),error,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12,markerfacecolor='red')
plt.xlabel('K-value')
plt.ylabel('Error_rate')
plt.title('Error_rate vs K_value curve')


# In[166]:


knn=KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
acc=accuracy_score(y_test,y_pred)


# In[167]:


acc*100


# In[168]:


knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
acc=accuracy_score(y_test,y_pred)


# In[169]:


acc*100


# In[170]:


knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
acc=accuracy_score(y_test,y_pred)




