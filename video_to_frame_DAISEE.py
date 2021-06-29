import os
import glob
import cv2

# Specify the file the videos are stored in
PathOut = r'C:/Users/chazzers/Desktop/DAiSEE/DataSet/Frames/'

# Specify the file you want the frames to be stored in
PathIn = r'C:/Users/chazzers/Desktop/DAiSEE/DataSet/Videos'

frame_count = 0
vid_count = 0

listing = os.listdir(PathIn)
for folder in listing:
    folder = r'C:/Users/chazzers/Desktop/DAiSEE/DataSet/Videos/' + folder
    for vid in os.listdir(folder):
        for video in os.listdir(folder + "/" + vid):
            print(video)
        # cap = cv2.VideoCapture(vid)
        # counter += 1
        # success = True

        # while success:
        #     success,image = cap.read()
        #
        #     if count % 15 == 0:
        #         cv2.imwrite(pathOut + 'video%d' % counter + 'frame%d.jpg' % count, image)
        #     count += 1

#     cap = cv2.VideoCapture(vid)
#     count = 0
#     counter += 1
#     success = True
#     while success:
#         success,image = cap.read()
# #         print('read a new frame:',success)
#         if count%60 == 0 :
#               cv2.imwrite(pathOut + 'video%d'%counter + 'frame%d.jpg'%count ,image)
#         count+=1
