import pandas as pd
import scipy.misc
import numpy as np
import os.path

df = pd.read_csv('fer2013.csv')



#images = df['pixels','Usage']

training = df.loc[df['Usage'] == 'Training']

images = training['pixels']
images = images.values

labels = training['emotion']
labels = labels.values

print(labels.shape)
print(labels)
np.save('labels.npy',labels)
'''
if not os.path.exists('images'):
    os.makedirs('images')
	
print(images.shape[0])
for i in range(0,images.shape[0]):
    image1 = np.fromstring(images[i],dtype=int,sep=" ")
    image1 = image1.reshape(48,48)
    scipy.misc.imsave('images/images{}.jpg'.format(i), image1)
'''