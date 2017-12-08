# Silvia Ionescu
# 12-7-2017

# AffectNet pre-processing. Extracts images from AffectNet files and saves them as a numpy array 

# AffectNet dataset = 414,799

# ('neutral:', 74874)
# ('happy:', 134416)
# ('sad:', 25459)
# ('surprise:', 14090)
# ('fear:', 6378)
# ('disgust:', 3803)
# ('anger:', 24882)
# ('contempt:', 3750)
# ('none:', 33088)
# ('uncertain:', 11645)
# ('non_face:', 82415)


# In[2]:


import pandas as pd
import numpy as np
import os

from PIL import Image
import cv2


# picks all 11 classifications
def add_image(classification, neutral, happy, sad, surprise, fear, disgust, anger, contempt, none, uncertain, non_face):
    #class_number = 4250
    class_number = 135000   
    add_img = False
    if classification == 0:
        neutral = neutral + 1
        if neutral <= class_number:
            add_img = True
    elif classification == 1:
        happy = happy + 1
        if happy <= class_number:
            add_img = True
    elif classification == 2:
        sad = sad + 1
        if sad <= class_number:
            add_img = True
    elif classification == 3:
        surprise = surprise + 1
        if surprise <= class_number:
            add_img = True       
    elif classification == 4:
        fear = fear + 1
        if fear <= class_number:
            add_img = True     
    elif classification == 5:
        disgust = disgust + 1
        if disgust <= class_number:
            add_img = True   
    elif classification == 6:
        anger = anger + 1
        if anger <= class_number:
            add_img = True  
    elif classification == 7:
        contempt = contempt + 1
        if contempt <= class_number:
            add_img = True   
    elif classification == 8:
        none = none + 1
        if none <= class_number:
            add_img = True 
    elif classification == 9:
        uncertain = uncertain + 1
        if uncertain <= class_number:
            add_img = True 
    elif classification == 10:
        non_face = non_face + 1
        if non_face <= class_number:
            add_img = True 

    return add_img, neutral, happy, sad, surprise, fear, disgust, anger, contempt, none, uncertain, non_face



# picks 8 classifications
def add_image_8class(classification, neutral, happy, sad, surprise, fear, disgust, anger, contempt, none, uncertain, non_face):
    #class_number = 4250
    class_number = 135000   
    add_img = False
    if classification == 0:
        neutral = neutral + 1
        if neutral <= class_number:
            add_img = True
    elif classification == 1:
        happy = happy + 1
        if happy <= class_number:
            add_img = True
    elif classification == 2:
        sad = sad + 1
        if sad <= class_number:
            add_img = True
    elif classification == 3:
        surprise = surprise + 1
        if surprise <= class_number:
            add_img = True       
    elif classification == 4:
        fear = fear + 1
        if fear <= class_number:
            add_img = True     
    elif classification == 5:
        disgust = disgust + 1
        if disgust <= class_number:
            add_img = True   
    elif classification == 6:
        anger = anger + 1
        if anger <= class_number:
            add_img = True  
    elif classification == 7:
        contempt = contempt + 1
        if contempt <= class_number:
            add_img = True   
    elif classification == 8:
        none = none + 1
    elif classification == 9:
        uncertain = uncertain + 1
    elif classification == 10:
        non_face = non_face + 1

    return add_img, neutral, happy, sad, surprise, fear, disgust, anger, contempt, none, uncertain, non_face

# extract images and save them into numpy arrays
if __name__ == '__main__':

    x = pd.read_csv('validation.csv')
    dataset_length = 4000
    
    data = x.values
    image_path = data[:,0]   
    face_x = data[:,1]
    face_y = data[:,2]
    face_width = data[:,3]
    face_height = data[:,4]
    y = data[:, 6]

    img_rows, img_cols = 48, 48 

    # pull classes into gray
    X_train = np.empty([dataset_length,img_rows, img_cols, 3])
    Y_train = np.empty([dataset_length, 1])
    
    neutral = 0
    happy = 0
    sad = 0
    surprise = 0
    fear = 0 
    disgust = 0
    anger = 0
    contempt = 0
    none = 0 
    uncertain = 0
    non_face = 0
    
    
    count = 0
    error_count = 0
    for i in range(0, len(y)):
        print("Loop:", i)
        
        add_img, neutral, happy, sad, surprise, fear, disgust, anger, contempt, none, uncertain, non_face = add_image_8class(y[i], neutral, happy, sad, surprise, fear, disgust, anger, contempt, none, uncertain, non_face)
        
        if add_img == True:
            path = image_path[i]
            array = path.split("/")
           
            pathname = os.path.join('AffectNet', array[0], array[1])
           
	    if os.stat(pathname).st_size == 0:
		error_count = error_count + 1;
		#print("error_count:", error_count)
		continue
            	
            Y_train[count] = y[i]

            print(pathname)

            img = cv2.imread(pathname)
            roi_gray = img[int(face_y[i]):int((face_y[i] + face_height[i])), int(face_x[i]):int((face_x[i]+face_width[i]))]
            
	    	# resize to 84x84
            test = cv2.resize(roi_gray, (img_rows, img_cols))
            
            X_train[count] = test
            print("count:", count)
            count = count + 1 

    #Y_train = np.expand_dims(y, axis=1)
   
    # prev. final_count = 45803

    print ("neutral:", neutral)
    print ("happy:", happy)
    print ("sad:", sad)
    print ("surprise:", surprise)
    print ("fear:", fear)
    print ("disgust:", disgust)
    print ("anger:", anger)

    print ("contempt:", contempt)
    print ("none:", none)
    print ("uncertain:", uncertain)
    print ("non_face:", non_face)

    print("Final count:", count)
    print("X_train shape:",X_train.shape)
    print("X_train shape:",Y_train.shape)
    np.save("X_valid_48x48x3_8class", X_train)
    np.save("Y_valid_48x48x3_8class", Y_train)



