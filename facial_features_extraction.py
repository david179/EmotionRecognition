#!/usr/bin/env python2
#
# OpenFace facial features extraction example
# Works on docker
#
#
import argparse
import cv2
import itertools
import os
import numpy as np
np.set_printoptions(precision=2)
import openface
import scipy.misc



# load input image from file
src = cv2.imread('trump.jpg',1)


# instantiate class extract landmarks with appropriate model
extract_landmarks = openface.AlignDlib('/root/openface/models/dlib/shape_predictor_68_face_landmarks.dat')

# get face bounding box
bounding_box = extract_landmarks.getLargestFaceBoundingBox(src,skipMulti=False)

landmarks = extract_landmarks.findLandmarks(src,bounding_box)


print len(landmarks)

# draw the landmarks in src
for i in range(len(landmarks)):
   cv2.circle(src,landmarks[i],4,(255,0,0),2)


cv2.imwrite('out_read.jpg',src)



