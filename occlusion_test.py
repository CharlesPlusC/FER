# %% Importing requirements
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from skimage import io, transform
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm
from matplotlib.colorbar import ColorbarBase
from matplotlib import cm
from dotenv import load_dotenv

from deepface.commons import functions, realtime, distance as dst
import pandas as pd
from PIL import Image
import imageio
load_dotenv()
# %%
# for Charles, pip install scikit-image

PATHOUT = os.getenv("PATH_OUT")
PATHIN = os.getenv("PATH_IN")
#object = pd.read_pickle(r'C:\Users\lizzy\OneDrive\Documents\Macbook Documents\COLLEGE\UCL\3rd year\Summer Project\DAiSEE_smol\Dataset\DataFrames\df0emotion_dfs.pkl')
REALPATH = os.getenv("VALIDATION")

def get_datagen(dataset):
    return ImageDataGenerator().flow_from_directory(
              dataset,
              target_size=(48,48),
              color_mode='grayscale',
              shuffle = True,
              class_mode='categorical',
              batch_size=32)

img_paths = []
for path in os.listdir(REALPATH):
    img_folder = REALPATH + path
    for face in os.listdir(img_folder):
        img_path = img_folder + '/' + face
        img_paths.append(img_path)
for i in img_paths:
    img_path = i
    out_path = i + '.tiff'
    detector_backend = 'opencv'
    enforce_detection = False
    img, region = functions.preprocess_face(img=img_path, target_size=(48, 48), grayscale=True,enforce_detection=enforce_detection, detector_backend=detector_backend, return_region=True)
    img.resize(48,48)
    cv2.imwrite(out_path,img)
    print("image done", img)

# for i in video_paths:
#     originalImage = cv2.imread(image)
#     grey_image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
#     grey_array = np.asarray(grey_image)
#     colour_to_grayscale.append(grey_array)


# X_test_gen = get_datagen(REALPATH)
#
# X_test = np.zeros((len(X_test_gen.filepaths), 48, 48, 1))
# Y_test = np.zeros((len(X_test_gen.filepaths), 7))
# for i in range(0,len(X_test_gen.filepaths)):
#   x = io.imread(X_test_gen.filepaths[i], as_gray=True) #loading the frames from the file location
#   X_test[i,:] = transform.resize(x, (48,48,1))
#   Y_test[i,X_test_gen.classes[i]] = 1
#
# #TODO Added model here in Stanford version
# #
# # Think it's making an array of the shape 40x40x1, full of 0.5s, then another two of smaller sizes.
# def iter_occlusion(image, size=8):
#     occlusion = np.full((size * 5, size * 5, 1), [0.5], np.float32)
#     occlusion_center = np.full((size, size, 1), [0.5], np.float32)
#     occlusion_padding = size * 2
# #
# # Padding the array/tensor adds 0s to convert to a shape that's better for the convolution, without losing any pixel info
#     print('padding...')
#     image_padded = np.pad(image, ( \
#                         (occlusion_padding, occlusion_padding), (occlusion_padding, occlusion_padding), (0, 0) \
#                         ), 'constant', constant_values = 0.0)
#
#     for y in range(occlusion_padding, image.shape[0] + occlusion_padding, size):
#         for x in range(occlusion_padding, image.shape[1] + occlusion_padding, size):
#             tmp = image_padded.copy()
#
#             tmp[y - occlusion_padding:y + occlusion_center.shape[0] + occlusion_padding, \
#                 x - occlusion_padding:x + occlusion_center.shape[1] + occlusion_padding] \
#                 = occlusion
#
#             tmp[y:y + occlusion_center.shape[0], x:x + occlusion_center.shape[1]] = occlusion_center
#
#             yield x - occlusion_padding, y - occlusion_padding, \
#                   tmp[occlusion_padding:tmp.shape[0] - occlusion_padding, occlusion_padding:tmp.shape[1] - occlusion_padding]
#
# # #######
# #
# i = 1
# data = X_test[i]
# correct_class = np.argmax(Y_test[i])
#
# # input tensor for model.predict
# inp = data.reshape(1,48,48,1)
# # image data for matplotlib's imshow
# img = data.reshape(48,48)
# # occlusion
# img_size = img.shape[0]
# occlusion_size = 4
# _ = plt.imshow(img,cmap='gray')
# cv2.imwrite(r'C:\Users\lizzy\OneDrive\Documents\Macbook Documents\COLLEGE\UCL\3rd year\Summer Project\DAiSEE_smol\Dataset\Next_Dataset\0 angry\actual_greyscale', img)
#
# # ## ADDING THE MODEL
# model = load_model('C:/Users/lizzy/OneDrive/Documents/Macbook Documents/COLLEGE/UCL/3rd year/Summer Project/DAiSEE_smol/emotionmodel.h5')
# #
# print('occluding...')
#
#     heatmap = np.zeros((img_size, img_size), np.float32)
#     class_pixels = np.zeros((img_size, img_size), np.int16)
#
#     counters = defaultdict(int)
#
#     for n, (x, y, img_float) in enumerate(iter_occlusion(data, size=occlusion_size)):
#         X = img_float.reshape(1,48,48,1)
#         out = model.predict(X)
#         print('#{}: {} @ {} (correct class: {})'.format(n, np.argmax(out), np.amax(out), out[0][correct_class]))
#         #print('x {} - {} | y {} - {}'.format(x, x + occlusion_size, y, y + occlusion_size))
#
#         heatmap[y:y + occlusion_size, x:x + occlusion_size] = out[0][correct_class]
#         class_pixels[y:y + occlusion_size, x:x + occlusion_size] = np.argmax(out)
#         counters[np.argmax(out)] += 1
#
#     pred = model.predict(inp)
#     print('Correct class: {}'.format(correct_class))
#     print('Predicted class: {} (prob: {})'.format(np.argmax(pred), np.amax(out)))
#
#     print('Predictions:')
#     for class_id, count in counters.items():
#         print('{}: {}'.format(class_id, count))
#
#     # Reverse heatmap so that red means important, blue means not
#     heatmap=1-heatmap
#
#     # displaying the occlusion map
#
#     # fig = plt.figure(figsize=(8, 8))
#     #
#     # ax1 = plt.subplot(1, 2, 1, aspect='equal')
#     # hm = ax1.imshow(heatmap)
#     #
#     # ax2 = plt.subplot(1, 2, 2, aspect='equal')
#     #
#     #
#     # vals = np.unique(class_pixels).tolist()
#     # bounds = vals + [vals[-1] + 1]  # add an extra item for cosmetic reasons
#     #
#     # custom = cm.get_cmap('Greens', len(bounds)) # discrete colors
#     #
#     # norm = BoundaryNorm(bounds, custom.N)
#     #
#     # cp = ax2.imshow(class_pixels, norm=norm, cmap=custom)
#     #
#     # divider = make_axes_locatable(ax1)
#     # cax1 = divider.append_axes("right", size="5%", pad=0.05)
#     # cbar1 = plt.colorbar(hm, cax=cax1)
#     #
#     # divider = make_axes_locatable(ax2)
#     # cax2 = divider.append_axes("right", size="5%", pad=0.05)
#     # cbar2 = ColorbarBase(cax2, cmap=custom, norm=norm,
#     #                          # place the ticks at the average value between two entries
#     #                          # e.g. [280, 300] -> 290
#     #                          # so that they're centered on the colorbar
#     #                          ticks=[(a + b) / 2.0 for a, b in zip(bounds[::], bounds[1::])],
#     #                          boundaries=bounds, spacing='uniform', orientation='vertical')
#     #
#     # cbar2.ax.set_yticklabels([n for n in np.unique(class_pixels)])
#     #
#     # fig.tight_layout()
#
#     plt.figure(figsize=(6, 6))
#
#     plt.imshow(img, cmap=cm.gray)
#     plt.pcolormesh(heatmap, cmap=plt.cm.jet, alpha=0.50)
#     plt.colorbar().solids.set(alpha=1)
#
# plt.show()
