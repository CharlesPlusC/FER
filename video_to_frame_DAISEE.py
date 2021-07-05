# importing dependencies
import os
import glob
import cv2
import pandas as pd
from deepface import DeepFace

# TODO: make all of these into functions

# ----------------------------------------------------------------------------#
# SPLITTING VIDEOS INTO FRAMES#

def Split_v2f(PathIn = r'C:/Users/lizzy/OneDrive/Documents/Macbook Documents/COLLEGE/UCL/3rd year/Summer Project/DAiSEE_smol/Dataset/Videos/', PathOut = r'C:/Users/lizzy/OneDrive/Documents/Macbook Documents/COLLEGE/UCL/3rd year/Summer Project/DAiSEE_smol/Dataset/Frames/',required_frame_rate = 2 ):


 # The frame rate that the film is recorded at -> Dependent on camera (usually 30)
 video_frame_rate = 30

 # Making a blank array that will be populated with the full paths of all videos
 video_paths = []

 # # Finding the name of all the video paths in the provided file structure
 for filename in os.listdir(PathOut):
     if filename.endswith('.jpg'):
         print("Are you sure? The videos seem to have already been split.")
         break   
 else:
 # do stuff if a file .true doesn't exist.
  for folder in os.listdir(PathIn):
    folder = PathIn + "/" + folder

    for vid in os.listdir(folder):
        vid = folder + "/" + vid

        for video in os.listdir(vid):
            video = vid + "/" + video
        video_paths.append(video)

 # using OpenCV to split all the videos specified into their component frames
  vid_count = 1

  for i in video_paths:
         cap = cv2.VideoCapture(i)
         vid_count+=1
         success = True
         frame_count = 1 #reset frame count to 1 at the start of every new video
         while success:
             success, image = cap.read()
             print('read a new frame:',success)
             if frame_count %(video_frame_rate*required_frame_rate) == 0:
                 cv2.imwrite(PathOut + 'video%d' % vid_count + 'frame%d.jpg' % frame_count, image)
             frame_count += 1

print(Split_v2f())


# ----------------------------------------------------------------------------
#PUTTING THE FRAMES THROUGH DEEPFACE AND OUTPUTTING THEM AS PD DATAFRAMES

# making a loop that takes the frames from one video at a time, puts them into an array and passes them through deepface
video_counter = 1
array_counter = 1
img_array = []
dfs = []


# takes all the photos that contain the number of 'video_counter' and puts them through deepface
for i in range(0, len(video_paths), 1):
    for filename in glob.glob(PathOut + 'video%d' % i + 'frame*.jpg'):
        # Read in the relevant images
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    # Pass them through deepface
    face_FER = DeepFace.analyze(img_path=img_array, actions=['emotion'], enforce_detection=False)
    img_array = []
    data = face_FER
    print(data)
    # Turning arrays into pandas dataframes and labelling emotions

    emotions = set()
    # First we need to find out all unique emotions
    for key, value in data.items():
        for emotion in value['emotion'].keys():
            emotions.add(emotion)

    rows = []
    columns = ['instance'] + list(emotions)

    for key, value in data.items():
        rows.append([0] * len(columns))  # Start creating a new row with zeros

        key = key.split('_')[1]  # Get the 1,2,3 out of the instance
        rows[-1][0] = key
        for emotion, emotion_value in value['emotion'].items():
            rows[-1][columns.index(emotion)] = emotion_value  # place the emotion in the correct index

    df = pd.DataFrame(rows, columns=columns)
    df.set_index('instance', inplace=True)
    dfs.append(df)


