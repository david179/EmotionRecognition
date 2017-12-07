import pandas as pd
import scipy.misc
import numpy as np
import sys,re
import os.path
from os import listdir
from os.path import isfile, join

# update mypath with the path to the AU files
mypath = 'images'
df = pd.read_csv('modified_training_for_FER2013.csv')

values = df.values

image_path = values[:,1]
emotions = np.asarray(values[:,2])

words = [ x[x.find('/')+1:] for x in image_path ]

words = np.asarray(words)

print(words)



onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

em = [0]
for filename in onlyfiles:
    name = '{}.jpg'.format(filename[0:filename.find('_det_0')])
    index = np.where(words == name)
	
    em.append(emotions[index[0][0]])
    
del em[0]

em = np.asarray(em)

print(em)
print(em.shape)

np.save('emotion_labels.npy',em)

