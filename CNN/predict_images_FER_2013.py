
# Silvia Ionescu
# 12-7-2017

# This predicts emotions for RGB/grayscale images. 
# Images are read, resized, transformed to grayscale, and the emotion is classified using 
# the pre-trained FER-2013 weights.  
 

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
    
    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    emotion_pairs = dict([(0, 'Angry'), (1, 'Disgust'), (2, 'Fear'), (3, 'Happy'), (4, 'Sad'),(5, 'Surprise'), (6, 'Neutral')])
    
    img_rows, img_cols = 48, 48
    mean = 129.385
    std = 65.057
    
    
    
    # Load the weights
    
    model = fer2013vgg(train=False)
    
    folder = './data/Trump_cropped/images_0000'
    original_image_folder = './data/Trump/images_0000'
    output_folder = './data/Trump_classified/images_0000'
    
    #folder = './data/Hilary_cropped/images_0000'
    #original_image_folder = './data/Hilary/images_0000'
    #output_folder = './data/Hilary_classified/images_0000'
    
    for i in range(1200, 1300):
    
    	# read image Trump/Hillary
        img = cv2.imread(folder + str(i) +'_crop.jpg')
        
        # convert grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


		# resize images
        gray_resized = cv2.resize(gray, (img_rows, img_cols))
        gray_norm = (gray_resized-mean)/(std+1e-7)

        gray_resize = np.expand_dims(cv2.resize(gray_norm, (img_rows, img_cols)), axis=2)
        gray_resize = np.expand_dims(gray_resize, axis=0)
		gray_resize = gray_resize.astype('float32')

		# Predict emotion
        predicted_x = model.predict(gray_resize)
        image_prediction = np.argmax(predicted_x,1)
        emotion = emotion_pairs[image_prediction[0]]


		# Save image + emotion        
        original_image = cv2.imread(original_image_folder + str(i) +'.jpg')
        print(original_image.shape)
        print(emotion)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original_image, emotion,(10, 40), font, 1,(255,255,255),2,cv2.LINE_AA)
        
        cv2.imwrite( output_folder + str(i) +'_classified.jpg', original_image);
      
        
        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #cv2.imshow('image',original_image)
        #cv2.waitKey(0)

