
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils,to_categorical
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
np.random.seed(seed)


# load data
data=pd.read_csv(r'D:\Datasets\Digit Recognizer\train.csv')

y=data[['label']]

X=data.drop(['label'],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)# reshape to be [samples][pixels][width][height]






# normalize inputs from 0-255 to 0-1
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
   
	model.add(Dense(10, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model






# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model


features=pd.read_csv(r'C:\Users\user\Downloads\test.csv')

input_val=features.values
sc.transform(input_val)
input_val = input_val.reshape(input_val.shape[0], 1, 28, 28).astype('float32')

pred_val=model.predict_classes(input_val)
pred_val.reshape((28000,1))
pred_val.astype(int)
#np.savetxt("ex1.csv", pred_val, delimiter=",")


