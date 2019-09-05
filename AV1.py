import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r'C:\Users\user\Downloads\AV1\train_LZdllcl.csv')

data.drop(['employee_id','region'],axis=1,inplace=True)

Dep=pd.get_dummies(data['department'],drop_first=True)
Sex=pd.get_dummies(data['gender'],drop_first=True)
Edu=pd.get_dummies(data['education'],drop_first=True)
Recruit=pd.get_dummies(data['recruitment_channel'],drop_first=True)

data=pd.concat([data,Dep,Sex,Edu,Recruit],axis=1)

data.drop(['department','gender','education','recruitment_channel'],axis=1,inplace=True)

data.fillna(value=data.loc[:,"previous_year_rating"].mean(),inplace=True)

X=data.drop(['is_promoted'],axis=1)
y=data[['is_promoted']]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense,Activation


classifier=Sequential()

classifier.add(Dense(32, input_dim=20))
classifier.add(Activation('relu'))

classifier.add(Dense(32))
classifier.add(Activation('relu'))

classifier.add(Dense(30))
classifier.add(Activation('relu'))


classifier.add(Dense(1))
classifier.add(Activation('sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


classifier.fit(X_train,y_train,batch_size=10,epochs=80)

features=pd.read_csv(r'C:\Users\user\Downloads\AV1\test_2umaH9m.csv')

Dep1=pd.get_dummies(features['department'],drop_first=True)
Sex1=pd.get_dummies(features['gender'],drop_first=True)
Edu1=pd.get_dummies(features['education'],drop_first=True)
Recruit1=pd.get_dummies(features['recruitment_channel'],drop_first=True)

features=pd.concat([features,Dep,Sex,Edu,Recruit],axis=1)

features.drop(['department','gender','education','recruitment_channel','employee_id','region'],axis=1,inplace=True)
features=features.loc[:23489,:]
features.fillna(value=features.loc[:,"previous_year_rating"].mean(),inplace=True)

features.values

y_pred=classifier.predict(sc.transform(features))

for i in range(0,23490):
    if(y_pred[i]>0.5):
        y_pred[i]=1
    else:
        y_pred[i]=0
    

np.savetxt("AV1.csv", y_pred, delimiter=",")

