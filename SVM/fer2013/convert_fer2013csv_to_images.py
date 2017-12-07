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
