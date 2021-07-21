#
import os
import glob
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

###Pseudocode:###

#   1) find out at what stage Deepface extracts faces from the entire image
#   2) find out how big "features" are in most images
#   3) apply small squares (the size of the "features") in predetermined positions on all frames
#   a select number of videos (maybe 5 to start)
#   4) run all of those videos through get_emotion() and get_engagement()
#   5) What areas being covered translate to lower accuracy? Is there any significance? This will tell us what the model is "looking for"

###Actual Code###
###-----1) find out at what stage Deepface extracts faces from the entire image-----###

# testing the part where the image get cropped to the size of the face
img_array = []
frame = "C:/Users/chazzers/Desktop/DAiSEE_smol/DataSet/Frames/video1frame60.jpg"#import random frame

img = cv2.imread(frame) #this reads the image in from its path
height, width, layers = img.shape # this creates three integers: 640, 480, and 3
size = (width, height) # this creates a tuple with two integers: 640 and 480
img_array.append(img) #this image is still (640,480)
# Doesnt seem to be any cropping here^

# Pass them through deepface
face_FER = DeepFace.analyze(img_path=img_array, actions=['emotion'], enforce_detection=False)
#papers say CNNs require detect and align steps in which the face is cropped so checking DeepFace as I think the cropping
#probably occurs inside deepface

#From DeepFace.py: faces are extracted from each frame (function called detector_backend; line 782). There is a
# possibility to use different detectors (retinaface, mtcnn, opencv, dlib or ssd). The default one is opencv.
# The detector_backend(0 function returns a "detected" and "aligned" face in numpy format.

#From OpencvWrapper.py: 1)crops the image to size of the face. 2) puts the eyes at the same level in each image

###-----2) find out how big "features" are in most images-----###

###-----3) apply small squares (the size of the "features") in predetermined positions on all frames a select number of videos (maybe 5 to start)-----###
# Not sure how to go about applying the squares to the images? probably possible to do it in the OpenCVWrapper code by
# adding a step ?