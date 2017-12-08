
# Silvia Ionescu
# 12-7-2017

# This code takes in numpy arrays in the form image_height x image_weight x 1 
# The input has been extracted from the FER-2013 .csv and put into .npy arrays with the above format.
# This code trains a VGG16 network using ~28k FER-2013 dataset images as training dataset and ~3.5k FER-2013 
# images as test. 
# Once it finishes the training, the network weights and pickle files are saved.   


from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.callbacks import History 

import pickle

class fer2013vgg:
    def __init__(self,train=True):
        self.num_classes = 7
        self.weight_decay = 0.0005

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('fer2013vgg.h5')

	# Build VGG16 model using Keras
	# Build the network of vgg for 8 classes
    def build_model(self):
         

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same', , activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def train(self,model):

        # Set training parameters
        batch_size = 128
        maxepoches = 150
        learning_rate = 0.01
        lr_decay = 1e-6
        
        img_rows, img_cols = 48, 48

        # Load the FER-2013 pre-processed dataset
      
        x = np.load('./facial_data_X.npy')
        y = np.load('./facial_labels.npy')
        
        # Normalize
        x -= np.mean(x, axis=0)
        x /= np.std(x, axis=0)
        
        # Split the dataset into test and training splits
        X_train = x[0:28710,:]
        x_train = X_train.reshape((X_train.shape[0], img_rows, img_cols, 1))
        y_train = y[0:28710]
        
        print(x_train.shape)
        print(y_train.shape)
        
        X_crossval = x[28710:32300,:]
        x_test = X_crossval.reshape((X_crossval.shape[0], img_rows, img_cols, 1))
        y_test = y[28710:32300]
        
        print(x_test.shape)
        print(y_test.shape)
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

		# Make hot-one vectors
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        lrf = learning_rate


        #data augmentation
        # set input mean to 0 over the dataset
        datagen = ImageDataGenerator(
            featurewise_center=False,  
            samplewise_center=False,  
            featurewise_std_normalization=False,  
            samplewise_std_normalization=False,  
            zca_whitening=False,  
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False) 
        datagen.fit(x_train)

        #optimization details
        sgd = optimizers.SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


        # set pickle arrays

        test_acc = []
        test_loss = []
        train_acc = []
        train_loss = []
		
        
        for epoch in range(1,maxepoches):

            history = model.fit_generator(datagen.flow(x_train, y_train,
                                             batch_size=batch_size),
                                steps_per_epoch=x_train.shape[0] // batch_size,
                                epochs=epoch,
                                validation_data=(x_test, y_test),initial_epoch=epoch-1)
 
            print(history.history.keys())
            test_acc.append(history.history['val_acc'])
            test_loss.append(history.history['val_loss'])
            train_loss.append(history.history['loss'])
            train_acc.append(history.history['acc'])
            #print("Test accuracy:", test_acc)
            #print("Train accuracy:", train_acc)
	    
            pickle.dump(test_acc, open("fer2013_test_acc_150epoch_lr01.p","wb"))
            pickle.dump(test_loss, open("fer2013_test_loss_150epoch_lr01.p","wb"))
            pickle.dump(train_acc, open("fer2013_train_acc_150epoch_lr01.p","wb"))
            pickle.dump(train_loss, open("fer2013_train_loss_150epoch_lr01.p","wb"))

			# save weights
            model.save_weights('fer2013vgg_150epoch_lr01.h5')

        return model

if __name__ == '__main__':

    print("Run: 150 epoch, learning rate = 0.01")
    model = fer2013vgg()

