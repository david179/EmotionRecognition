import time
import argparse
import os
import sys,re
import numpy as np
from os import listdir
from os.path import isfile, join
import _pickle as cPickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2


def train(epochs=10000, random_state=0,kernel='rbf', decision_function='ovr', train_model=True,label_path=""):
        
        if train_model:
        
            print ("loading KDFE dataset AUs")
            
            # intensities
            i_kdfe = np.load('intens_kdfe.npy')
            # occurences
            o_kdfe = np.load('occur_kdfe.npy')
            # emotion labels
            label_kdfe = np.ravel(np.load('emotions_kdfe.npy'))
            
            
            # intensities + occurences
            total_kdfe = np.concatenate((i_kdfe,o_kdfe),axis=1)
            
            print('intensities')
            print(i_kdfe.shape)
            print('occurences')
            print(o_kdfe.shape)
            print('labels')
            print(label_kdfe.shape)
           
            # create train and validation data
            i_kdfe_train = i_kdfe[500:]
            i_kdfe_val = i_kdfe[0:500]
            
            o_kdfe_train = o_kdfe[500:]
            o_kdfe_val = o_kdfe[0:500]
            
            total_kdfe_train = total_kdfe[500:]
            total_kdfe_val = total_kdfe[0:500]
            
            label_kdfe_train = label_kdfe[500:]
            label_kdfe_val = label_kdfe[0:500]
           
            
            print('\nImages per emotion')
            print('Angry: {}'.format(label_kdfe[label_kdfe == 0].shape[0]))
            print('Disgust: {}'.format(label_kdfe[label_kdfe == 1].shape[0]))
            print('Fear: {}'.format(label_kdfe[label_kdfe == 2].shape[0]))
            print('Happy: {}'.format(label_kdfe[label_kdfe == 3].shape[0]))
            print('Sad: {}'.format(label_kdfe[label_kdfe == 4].shape[0]))
            print('Surprise: {}'.format(label_kdfe[label_kdfe == 5].shape[0]))
            print('Neutral: {}'.format(label_kdfe[label_kdfe == 6].shape[0]))
            
            
            train_data = total_kdfe_train
            train_label = label_kdfe_train
            
            val_data = total_kdfe_val
            val_label = label_kdfe_val
               
        
            # Training phase
            print ("building model")
            model = SVC(random_state=random_state, max_iter=epochs, kernel=kernel, decision_function_shape=decision_function)

            print ("Training")
            print ("kernel: {}".format(kernel))
            print ("decision function: {} ".format(decision_function))
            print ("max epochs: {} ".format(epochs))
            print( "")
            print ("Training samples: {}".format(train_data.shape[0]))
            print( "Validation samples: {}".format(val_data.shape[0]))
            print( "")
            start_time = time.time()
            
            model.fit(train_data,train_label)
            
            training_time = time.time() - start_time
            print ("training time = {0:.1f} sec".format(training_time))

            print ("\nsaving model")
            with open('trained_model.bin', 'wb') as f:
                    cPickle.dump(model, f)

            print ("\nevaluating")
            validation_accuracy = evaluate(model,val_data,val_label)
            print ("  - validation accuracy = {0:.1f}".format(validation_accuracy*100))
            
            return validation_accuracy
            
        else:
        
            # Testing phase : load saved model and evaluate on test data
            print("parse test images AUs")
            
            mypath = 'predict_images_AU'
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

            final_intens = np.empty([17])
            final_occur = np.empty([18])
        
            image_names = ['']
            
            for filename in onlyfiles:

                if os.path.exists('{}/{}'.format(mypath,filename)):
                
                    data = open('{}/{}'.format(mypath,filename),'r').read()

                    name = '{}.jpg'.format(filename[0:filename.find("_det_0")])
                    
                    image_names.append(name)
                    
                    intens = data[data.find("intensities")+len("intensities: 17 {"):data.find('}\nau occ')]
                    occur =  data[data.find("occurences:")+len("occurences: 18 {"):]

                    intens = re.split('[ \n]',intens)
                    intens = intens[1:len(intens)-1]
                    intens = intens[1::2]
                    intens = list(map(float, intens))

                    occur = re.split('[ \n]',occur)
                    occur = occur[1:len(occur)-2]
                    occur = occur[1::2]
                    occur = list(map(float, occur))
                    
                    intens = np.array(intens)
                    occur = np.array(occur)
                    
                    final_intens = np.vstack((final_intens,intens))
                    final_occur = np.vstack((final_occur,occur))
                    
                    
                else:
                    continue
                    
            final_intens = np.delete(final_intens,0,0)
            final_occur = np.delete(final_occur,0,0)
            
            del image_names[0]
            
            # Optional save the AUs 
            #np.save('predict_intens.npy',final_intens)
            #np.save('predict_occur.npy',final_occur)
            
            
            print ("loading pretrained model")   
            if os.path.isfile('trained_model.bin'):
                with open('trained_model.bin', 'rb') as f:
                        model = cPickle.load(f)
            else:
                print ("Error: file '{}' not found".format('trained_model.bin'))
                exit()

            print( "")
            print ("Number of images: {}".format(final_intens.shape[0]))
            print ("")
            print ("Predicting")
            start_time = time.time()
            
           
            # concatenate intensities and occurences
            val = np.concatenate((final_intens, final_occur), axis=1)
            
            em = dict([(0,'Angry'),(2,'Afraid/Fear'),(1,'Disgusted'),(3,'Happy'),(6,'Neutral'),(4,'Sad'),(5,'Surprised')])
            
            if label_path == "":
                predicted_Y = model.predict(val)
                index = 0
                cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 600,600)
                for i in predicted_Y:
                    image = cv2.imread('predict_images/{}'.format(image_names[index]),1)
                    cv2.putText(image,em[i],(40,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255)) 
                    cv2.imshow('image',image)
                    cv2.waitKey(0)
                    index += 1
                    print(em[i])
                
            else:
                print('loading lables from: {}'.format(label_path))
                predict_images_labels = np.load(label_path)
                accuracy = evaluate(model,val,predict_images_labels)
                
                print('accuracy: {}'.format(accuracy*100))
                
            return 

def evaluate(model, X, Y):

        predicted_Y = model.predict(X)
            
        accuracy = accuracy_score(Y, predicted_Y)
        return accuracy

# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", default="no", help="if 'yes', launch training from command line")
parser.add_argument("-e", "--evaluate", default="no", help="if 'yes', launch evaluation on test dataset")
parser.add_argument("-l", "--labels", default="", help="path to emotion labels of images in predict folder")
args = parser.parse_args()
if args.train=="yes" or args.train=="Yes" or args.train=="YES":
        train()
if (args.evaluate=="yes" or args.evaluate=="Yes" or args.evaluate=="YES") and args.labels =="":
    train(train_model=False)
else:
    train(train_model=False,label_path=args.labels)