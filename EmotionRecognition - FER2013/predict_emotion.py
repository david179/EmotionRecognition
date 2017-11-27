import numpy as np

import theano
import os
from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape
from keras.models import Sequential
from keras.utils import np_utils

import tensorflow as tf

tf.python.control_flow_ops = tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
import numpy
import csv
import scipy.misc
import scipy
from scipy import ndimage
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


img_rows, img_cols = 48, 48
model = Sequential()
model.add(Convolution2D(64, 5, 5, border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='th'))
model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th')) 
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th')) 
model.add(Convolution2D(64, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(Convolution2D(128, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(Convolution2D(128, 3, 3))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

model.add(Flatten())
model.add(Dense(1024))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(Dropout(0.2))


model.add(Dense(7))


model.add(Activation('softmax'))


ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer=ada,
              metrics=['accuracy'])
model.summary()


import cv2

mat = cv2.imread('t3.jpg',0)

# resize mat to 48x48
resized = cv2.resize(mat,(48,48))

normalized = resized - np.mean(resized, axis=0)
normalized /= np.std(normalized, axis=0)


reshaped = normalized.reshape(1,1,48,48)

model.load_weights("Model.29-0.4390.hdf5")


print(reshaped)

pred = model.predict(reshaped,batch_size=5,verbose=0)

print(pred)

from keras import backend as K

print('0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral')
for ix in range(1):
    print(np.argmax(pred[ix]))
	
cv2.waitKey(0)

