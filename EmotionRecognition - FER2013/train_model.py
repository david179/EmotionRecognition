
# coding: utf-8

# In[2]:


import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import UpSampling2D, ZeroPadding2D, Reshape

from keras.utils import np_utils

from keras.optimizers import Adadelta
from keras.regularizers import l2

import scipy.misc
import scipy
from scipy import ndimage
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


from keras import backend as K

import tensorflow as tf


# In[3]:


x = np.load('./facial_data_X.npy')
y = np.load('./facial_labels.npy')
print (x.shape)
x -= np.mean(x, axis=0)
x /= np.std(x, axis=0)


# In[4]:


img_rows, img_cols = 48, 48

X_train = x[0:28710,:]
Y_train = y[0:28710]
print(X_train.shape , Y_train.shape)
X_crossval = x[28710:32300,:]
Y_crossval = y[28710:32300]
print (X_crossval.shape , Y_crossval.shape)


# In[5]:


X_train = X_train.reshape((X_train.shape[0], img_rows, img_cols, 1))
X_crossval = X_crossval.reshape((X_crossval.shape[0], img_rows, img_cols, 1))
print(X_train.shape)
print(X_crossval.shape)


# In[6]:


img_rows, img_cols = 48, 48
input_shape = (img_rows, img_cols, 1)

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), padding='valid', input_shape=input_shape))
model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2)))
model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1))) 
model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1))) 
model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', weights=None))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(128, kernel_size=(3, 3)))
model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
model.add(Conv2D(128, kernel_size=(3, 3)))
model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', weights=None))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', weights=None))
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', weights=None))
model.add(Dropout(0.2))


model.add(Dense(7))


model.add(Activation('softmax'))

ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer=ada,
              metrics=['accuracy'])
          
model.summary() 


# In[7]:


print(y.shape)
y_ = np_utils.to_categorical(y, num_classes=7)

print(y_.shape)
Y_train = y_[:28710]
Y_crossval = y_[28710:32300]
print(X_crossval.shape, model.input_shape, Y_crossval.shape)


# In[ ]:



datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
filepath='Model.{epoch:02d}-{val_acc:.4f}.hdf5'
checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

model.fit_generator(datagen.flow(X_train, Y_train,
                    batch_size=128),
                    epochs=30,
                    validation_data=(X_crossval, Y_crossval),
                    steps_per_epoch=X_train.shape[0], callbacks=[checkpointer])

