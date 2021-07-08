# importing dependencies
import os
import glob
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

# ----------------------------------------------------------------------------#
###SPLITTING VIDEOS INTO FRAMES FOR ANALYSIS###

# Specify Paths#
DATAFRAMESOUT = "C:/Users/ccons/OneDrive/Desktop/DAiSEE_smol/Dataset/DataFrames/"
# Specify the file you want the frames to be stored in
PATHOUT = r'C:/Users/ccons/OneDrive/Desktop/DAiSEE_smol/Dataset/Frames/'
# Specify the file the videos are stored in
PATHIN = r'C:/Users/ccons/OneDrive/Desktop/DAiSEE_smol/Dataset/Videos/'

# Specify Variables#
# The frame rate that the film is recorded at -> Dependent on camera (usually 30)
video_frame_rate = 30

# The frame rate we want (i.e. "I want a frame every x seconds")
required_frame_rate = 2

# Making a blank array that will be populated with the full paths of all videos
video_paths = []

# Finding the name of all the video paths in the provided file
# for folder in os.listdir(PATHIN):
#     folder = PATHIN + folder
#
#     for vid in os.listdir(folder):
#         vid = folder + "/" + vid
#
#         for video in os.listdir(vid):
#             video = vid + "/" + video
#         video_paths.append(video)
#
# # using OpenCV to split all the videos specified into their component frames
# vid_count = 1

# for i in video_paths:
#     cap = cv2.VideoCapture(i)
#     vid_count+=1
#     success = True
#     frame_count = 1 #reset frame count to 1 at the start of every new video
#     while success:
#         success, image = cap.read()
#         print('read a new frame:',success)
#         if frame_count %(video_frame_rate*required_frame_rate) == 0:
#             cv2.imwrite(PATHOUT + 'video%d' % vid_count + 'frame%d.jpg' % frame_count, image)
#         frame_count += 1
#
# TODO: make this code not end with an error


# # # ----------------------------------------------------------------------------#
# PUTTING THE FRAMES THROUGH DEEPFACE AND OUTPUTTING THEM AS PD DATAFRAMES#
# making a loop that takes the frames from one video at a time, puts them into an array and passes them through deepface
video_counter = 1
array_counter = 1
img_array = []
dfs = []

# TODO: for some reason starting this loop at 0 or 1 gives me empty frames? maybe to do with the video counter
#  starting at 1?
for i in range(2, 10, 1):
    for filename in glob.glob(PATHOUT + 'video%d' % i + 'frame*.jpg'):
        print(filename)
        # Read in the relevant images
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    # Pass them through deepface
    face_FER = DeepFace.analyze(img_path=img_array, actions=['emotion'], enforce_detection=False)
    img_array = []
    data = face_FER
    # Turning arrays into pandas dataframes and labelling emotions

    emotions = set()
    # First we need to find out all unique emotions
    for key, value in data.items():
        for emotion in value['emotion'].keys():
            emotions.add(emotion)

    rows = []
    columns = ['vid%d' % i + 'instance'] + list(emotions)

    for key, value in data.items():
        rows.append([0] * len(columns))  # Start creating a new row with zeros

        key = key.split('_')[1]  # Get the 1,2,3 out of the instance
        rows[-1][0] = key
        for emotion, emotion_value in value['emotion'].items():
            rows[-1][columns.index(emotion)] = emotion_value  # place the emotion in the correct index

    df = pd.DataFrame(rows, columns=columns)
    df.set_index('vid%d' % i + 'instance', inplace=True)
    dfs.append(df)

# ----------------------------------------------------------------------------#
# TREATING THE DATA IN THE DATAFRAMES TO GET "ENGAGEMENT"

# TODO: currently applying to all frames; make it so that we can split frames belonging to different individuals

# Getting averages and rolling averages of positive-valence, negative-valence, and neutral emotions
for df in dfs:

    # average of negative and positive valence emotions, and neutral
    df['neg_valence_avg'] = np.mean(df[['fear', 'disgust', 'angry', 'sad']], axis=1)
    df['pos_valence_avg'] = np.mean(df[['happy', 'surprise']], axis=1)
    df['neutral_avg'] = np.mean(df[['neutral']], axis=1)

    # Taking a rolling average of these (length of the rolling average = 2% the length of the dataframe(or 2 frames
    # whichever is biggest))
    while int(len(df) * 0.02) > 1:
        three_percent_len = int(len(df) * 0.02)
    else:
        three_percent_len = 1

    df['neg_valence_avg_roll'] = df['neg_valence_avg'].rolling(window=three_percent_len).mean()
    df['pos_valence_avg_roll'] = df['pos_valence_avg'].rolling(window=three_percent_len).mean()
    df['neutral_avg_roll'] = df['neutral_avg'].rolling(window=three_percent_len).mean()

# Making a dataframe that compares all the videos to each other (no longer computing intra-video stats but inter-video)

# empty arrays to add inter-video analysis data
valence_per_vid = []
variance_per_vid = []
total_vid_variance = []

# list of median of positive emotions for each video
for df in dfs:
    for i in df:
        # list of median of pos,neg and neutral emotions for each video (one value for each video), and length of video
        valence_values = [(df['neg_valence_avg'].median()), df['pos_valence_avg'].median(),
                          df['neutral_avg'].median(), len(df)]
    variance_per_vid.append(df.iloc[:, 0:7].var())  # variance for each emotion in a video; Appended to a list
    print(variance_per_vid)
    # append these values to lists of lists
    valence_per_vid.append(valence_values)

###VALENCE###
# turning list of lists into a dataframe
video_valence_df = pd.DataFrame(valence_per_vid, columns=["neg_avg_vid", "pos_avg_vid", "neutral_avg_vid", "vid_len"])
# Average positive and negative valence across all videos
total_vid_pos = (video_valence_df["pos_avg_vid"].mean())
total_vid_neg = (video_valence_df["neg_avg_vid"].mean())
video_valence_df["total_vid_pos"] = total_vid_pos
video_valence_df["total_vid_neg"] = total_vid_neg

###VARIANCE###
# valence for each emotion group and video length for each video
video_variance_df = pd.DataFrame(variance_per_vid)  # variance for each emotion in each video
# average variance of all emotions in any video (except neutral)
all_vid_variance = video_variance_df[['happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']].mean(axis=1)
all_vid_variance_df = pd.DataFrame(all_vid_variance, columns=["variance_per_video"])
# average variance across all videos
total_vid_variance = (all_vid_variance.mean())
all_vid_variance_df["var_avg_all_vids"] = total_vid_variance

###VARIANCE + VALENCE###
# merging the frames containing data on variance, valence and video length
video_stats_df = pd.merge(all_vid_variance_df, video_valence_df, left_index=True, right_index=True)
# TODO: fix this merge so that it does not merge on index. Need to add video_name as a column to both datasets and
#  merge using that column. Sometimes index does weird things and we will have no way of knowing if it goes wrong.

###"ENGAGEMENT"###
# Getting difference between average negative score for all videos and average negative score for each video
video_stats_df["pos_diff"] = video_stats_df["pos_avg_vid"] - video_stats_df["total_vid_pos"]
video_stats_df["neg_diff"] = video_stats_df["neg_avg_vid"] - video_stats_df["total_vid_neg"]
video_stats_df["var_diff"] = video_stats_df["variance_per_video"] - video_stats_df["var_avg_all_vids"]

# Getting the differences between the highest and lowest values for emotions and variance
video_stats_df["variance_range"] = video_stats_df["var_diff"].max() - video_stats_df["var_diff"].min()
video_stats_df["pos_range"] = video_stats_df["pos_diff"].max() - video_stats_df["pos_diff"].min()
video_stats_df["neg_range"] = video_stats_df["neg_diff"].max() - video_stats_df["neg_diff"].min()

# Using the above ranges to calculate the percentage difference
video_stats_df["percent_diff_var"] = (video_stats_df["var_diff"] / video_stats_df["variance_range"]) * 100
video_stats_df["percent_diff_pos"] = (video_stats_df["pos_diff"] / video_stats_df["pos_range"]) * 100
video_stats_df["percent_diff_neg"] = (video_stats_df["neg_diff"] / video_stats_df["neg_range"]) * 100

# Exporting Frame to .pkl file#
print("", dfs)
print("video stats", video_stats_df)
video_stats_df.to_pickle(DATAFRAMESOUT + "cross_video_stats_df.pkl")
dfs.to_pickle(DATAFRAMESOUT + "emotion_dfs.pkl")

# TODO: add engagement scores to compare with from the DAISEE dataset
# TODO: make it so that the frame is somehow split by person
