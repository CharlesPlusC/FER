import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deepface import DeepFace
import warnings

warnings.filterwarnings("ignore")
from tensorflow.keras import backend as K
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

import imageio

from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID, DlibWrapper, ArcFace, Boosting
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst

import tensorflow as tf

tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
    import logging

    tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.disable_eager_execution()
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
# img_array = []
# frame = "C:/Users/chazzers/Desktop/DAiSEE_smol/DataSet/Frames/video1frame60.jpg" #import random frame
# #
# img = cv2.imread(frame) #this reads the image in from its path
# height, width, layers = img.shape # this creates three integers: 640, 480, and 3
# size = (width, height) # this creates a tuple with two integers: 640 and 480
# img_array.append(img) #this image is still (640,480)
# # Doesnt seem to be any cropping here^

# Pass them through deepface face_FER = DeepFace.analyze(img_path=img_array, actions=['emotion'],
# enforce_detection=False) papers say CNNs require detect and align steps in which the face is cropped so checking
# DeepFace as I think the cropping probably occurs inside deepface

# From DeepFace.py: faces are extracted from each frame (function called detector_backend; line 782). There is a
# possibility to use different detectors (retinaface, mtcnn, opencv, dlib or ssd). The default one is opencv.
# The detector_backend(0 function returns a "detected" and "aligned" face in numpy format. All faces "forced" to 48*48

# From OpencvWrapper.py: 1)crops the image to size of the face. 2) puts the eyes at the same level in each image

###-----2) find out how big "features" are in most images-----###

###-----3) apply small squares (the size of the "features") in predetermined positions on all frames a select number of videos (maybe 5 to start)-----###
# Not sure how to go about applying the squares to the images? probably possible to do it in the OpenCVWrapper code by
# adding a step ?
# after some digging seems like we could change the analyze() function in DeepFace.py (line 377 does emotion), to include
# mechanism of adding a rectangle?

#
# #This is how you add a rectangle to an image:
# img_array = []
# frame = "C:/Users/chazzers/Desktop/DAiSEE_smol/DataSet/Frames/video1frame60.jpg" #import random frame
# #
# img = cv2.imread(frame) #this reads the image in from its path
# height, width, layers = img.shape # this creates three integers: 640, 480, and 3
# size = (width, height) # this creates a tuple with two integers: 640 and 480
# img_array.append(img) #this image is still (640,480)
#
# cv2.rectangle(img, pt1=(200,200), pt2=(300,300), color=(0,0,0), thickness=-1)
#
# plt.imshow(img)
# plt.show()

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)
    for var, grad in zip(var_list, grads)]

# build the emotion model
model = DeepFace.build_model('Emotion')

photo = cv2.imread('C:/Users/chazzers/Desktop/DAiSEE_smol/DataSet/Frames/video1frame60.jpg')
input_img = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
img_width = 640
img_height = 480

#
layer_dict = dict([(layer.name, layer) for layer in model.layers])

layer_name = 'conv2d_3'
filter_index = 126  # can be any integer from 0 to 511, as there are 512 filters in that layer

# build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer_output = layer_dict[layer_name].output
loss = K.mean(layer_output[:, :, :, filter_index])

grads = K.gradients(loss, input_img)[0]

input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 128.
imageio.imwrite('input_img_data.png', input_img_data)

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

img = input_img_data[0]
img = deprocess_image(img)

imageio.imwrite('%s_filter_%d.png' % (layer_name, filter_index), img)