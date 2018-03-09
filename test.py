import sys, os
import google.protobuf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

import keras
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Activation, Dropout, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam, SGD

#read input and set label 
data_train = pd.read_csv('./input/training.csv')
labels = data_train.filter(['Label'], axis=1)
labels.replace({'s':1, 'b':0}, inplace=True)
Y_train = labels.values.astype('int32')
Y_train = np_utils.to_categorical(Y_train) 
#print labels

#drop phi and label features
col_names = list(data_train)
for names in col_names:
  if 'phi' in names:
    #print names
    data_train = data_train.drop(names, axis=1)
data_train = data_train.drop('Label', axis=1)
data_train.astype('float32')

#Standardization
scaler = StandardScaler()
scaler.fit(data_train)
data_sc_train = scaler.transform(data_train)

#PCA transform
NCOMPONENTS = 26
pca = PCA(n_components=NCOMPONENTS)
data_pca_train = pca.fit_transform(data_sc_train)
pca_std = np.std(data_pca_train)
print(data_pca_train.shape)


#Keras
a = 500
b = 0.5
init = 'glorot_uniform'

inputs = Input(shape=(26,))
x = Dense(a, kernel_regularizer=l2(1E-2))(inputs)
x = BatchNormalization()(x)

branch_point1 = Dense(a, name='branch_point1')(x)

x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)

x = BatchNormalization()(x)
x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)

x = add([x, branch_point1])

x = BatchNormalization()(x)
branch_point2 = Dense(a, name='branch_point2')(x)

x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)
x = BatchNormalization()(x)
x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)

x = add([x, branch_point2])

x = BatchNormalization()(x)
x = Dense(a, activation='relu', kernel_initializer=init, bias_initializer='zeros')(x)
x = Dropout(b)(x)

predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0), metrics=['binary_accuracy'])
#sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
#model.compile( loss = "categorical_crossentropy", optimizer = sgd, metrics=['binary_accuracy'] )

model.save('model.h5')
model.summary()
history = model.fit(data_pca_train, Y_train, epochs=10, batch_size=500, validation_split=0.3)
#print(history.history.keys())
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('binary crossentropy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
