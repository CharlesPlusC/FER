#importing dependencies
import os
import glob
import cv2

# Specify the file the videos are stored in
PathOut = r'D:/DAiSEE/DAiSEE/DataSet/Frames/'

# Specify the file you want the frames to be stored in
PathIn = r'D:/DAiSEE/DAiSEE/DataSet/Videos/'
# changed path to D: instead of C: to test things
# TODO: change this to correct path when we want to process all the videos

# The frame rate that the film is recorded at -> Dependent on camera (usually 30)
video_frame_rate = 30

# The frame rate we want (i.e. "I want a frame every x seconds")
required_frame_rate = 5

# Making a blank array that will be populated with the full paths of all videos
video_paths = []

# Finding the name of all the video paths in the provided file structure

for folder in os.listdir(PathIn):
    folder = PathIn + folder

    for vid in os.listdir(folder):
        vid = folder + "/" + vid

        for video in os.listdir(vid):
            video = vid + "/" + video
        video_paths.append(video)

# using OpenCV to split all the videos specified into their component frames
vid_count = 1
frame_count = 1

for i in video_paths:
    cap = cv2.VideoCapture(i)
    success, image = cap.read()
    frame_count = 1

    while success:

        if frame_count % (video_frame_rate * required_frame_rate) == 0:
            cv2.imwrite(PathOut + 'video%d' % vid_count + 'frame%d.jpg' % frame_count, image)
            success, image = cap.read()
        frame_count += 1
    vid_count += 1
