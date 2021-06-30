import os
import glob
import cv2

# Specify the file the videos are stored in
PathOut = r'C:/Users/chazzers/Desktop/DAiSEE/DataSet/Frames/'

# Specify the file you want the frames to be stored in
PathIn = r'C:/Users/chazzers/Desktop/DAiSEE/DataSet/Videos/'



video_paths = []

for folder in os.listdir(PathIn):
    folder = PathIn + folder

    for vid in os.listdir(folder):
        vid = folder + "/" + vid

        for video in os.listdir(vid):
            video = vid + "/" + video
        video_paths.append(video)
# print(video_paths)

vid_count = 0
frame_count = 0

for i in video_paths:
    cap = cv2.VideoCapture(i)
    success, image = cap.read()
    vid_count += 1

    while success:

        if frame_count % 250 == 0:
            cv2.imwrite(PathOut + 'video%d' % vid_count + 'frame%d.jpg' % frame_count, image)
            success, image = cap.read()
        frame_count += 1
