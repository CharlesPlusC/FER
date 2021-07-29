# importing dependencies
import os
import glob
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt
from scipy import stats
#
# rng = np.random.default_rng()
# #
# #1) Plot variance of all videos vs all labels of all videos
# df = pd.read_pickle("C:/Users/chazzers/Desktop/DAiSEE_smol/DataSet/DataFrames/cross_video_stats_df.pkl")
# labels = pd.read_csv("C:/Users/chazzers/Desktop/DAiSEE_smol/DataSet/Labels/AllLabels.csv")
# vid_1_eng = labels.Engagement.iloc[0:113]
# vid_6_eng = labels.Engagement.iloc[282:360]
# vid_7_eng = labels.Engagement.iloc[361:502]
# vid_8_eng = labels.Engagement.iloc[503:511]
# vid_10_eng = labels.Engagement.iloc[512:512]
# vid_11_eng = labels.Engagement.iloc[513:587]
# vid_12_eng = labels.Engagement.iloc[588:674]
# vid_13_eng = labels.Engagement.iloc[675:682]
# vid_14_eng = labels.Engagement.iloc[683:771]
# vid_15_eng = labels.Engagement.iloc[772:849]
#
# engs = pd.concat([vid_1_eng ,vid_6_eng ,vid_7_eng ,vid_8_eng ,vid_10_eng,vid_11_eng,vid_12_eng,vid_13_eng,vid_14_eng,vid_15_eng])
# all_engs = engs.reset_index()
# all_vids = df.iloc[0:672]
# #
# # # # ### Shared x-axis graph###
# data1 = all_vids['percent_diff_var']
# # # #data1 = df['percent_diff_neg']
# # # #data1 = df['percent_diff_var']
# data2 = all_engs
# # # data2 = labels["Boredom"]
# # # # data2 = labels["Confusion"]
# #
# fig, ax1 = plt.subplots()
# #
# color1 = 'tab:red'
# ax1.set_xlabel('video',fontsize= 12)
# ax1.set_ylabel('% Difference in variance', color=color1,fontsize= 12)
# ax1.plot( data1, color=color1)
# ax1.tick_params(axis='y', labelcolor=color1)
# plt.yticks(fontsize= 12)
# plt.xticks(fontsize= 12)
# plt.ylim(-20,100)
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color2 = 'tab:blue'
# ax2.set_ylabel('Engagement score(/4)', color=color2,fontsize= 12)  # we already handled the x-label with ax1
# ax2.plot( data2, color=color2)
# ax2.tick_params(axis='y', labelcolor=color2)
# plt.yticks(fontsize= 12)
# plt.ylim(0,4)
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()


## linear regression graph###
# x = data1
# y = data2["Engagement"]
#
# res = stats.linregress(x, y)
# print(f"R-squared: {res.rvalue**2:.6f}")
#
# plt.plot(x, y, 'o', label='original data')
# plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
# plt.legend()
# plt.show()

# ###################### NOT MAKING
#
# DATAFRAMESOUT = os.getenv("DATA_FRAMES_OUT")
# PATHIN = os.getenv("PATH_IN")
# PATHOUT = os.getenv("PATH_OUT")
# img_array = []
# dfs = []
# video_paths = []
# video_names = []
# person_folder_names = []
# #
# # added the video counter here so it does not depend on the previous function
# video_counter = 0  # how many videos there are
# for i in os.listdir(PATHIN):
#     person_folder = PATHIN + i
#     for vid_folder in os.listdir(person_folder):
#         vid_folder = person_folder + "/" + vid_folder
#         for video in os.listdir(vid_folder):
#             video_names.append(video)  # saving just the name of the video into an array
#             video = vid_folder + "/" + video
#             video_paths.append(video)  # saving the whole path of the video into another array
#             person_folder_names.append(i)  # add an instance of 'folder name' for each video
#             video_counter += 1
# for i in range(0, video_counter, 1):
#     for filename in glob.glob(PATHOUT + 'video%d' % i + 'frame*.jpg'):
#         # Read in the relevant images
#         img = cv2.imread(filename)
#         height, width, layers = img.shape
#         size = (width, height)
#         img_array.append(img)
#     # Pass them through deepface
#     face_FER = DeepFace.analyze(img_path=img_array, actions=['emotion'], enforce_detection=False)
#     img_array = []  # reset image array to be blank
#     data = face_FER
#     # Turning arrays into pandas dataframes and labelling emotions
#     emotions = set()
#     # First we need to find out all unique emotions
#     for key, value in data.items():
#         for emotion in value['emotion'].keys():
#             emotions.add(emotion)
#     rows = []
#     columns = ['instance'] + list(emotions)
#     for key, value in data.items():
#         rows.append([0] * len(columns))  # Start creating a new row with zeros
#         key = key.split('_')[1]  # Get the 1,2,3 out of the instance
#         rows[-1][0] = key
#         for emotion, emotion_value in value['emotion'].items():
#             rows[-1][columns.index(emotion)] = emotion_value  # place the emotion in the correct index
#     df = pd.DataFrame(rows, columns=columns)
#     df.set_index('instance', inplace=True)
#     dfs.append(df)  # TODO: need to index this with 'video names'
# #
# index_arrays = (person_folder_names,video_names)
# dfs_index = pd.MultiIndex.from_arrays(index_arrays, names=["person", "video"])
# dfs_frame = pd.DataFrame(dfs,index = dfs_index, columns=['video_emotion']) #dataframe indexed by folder name and video name (two levels)
# dfs_frame.to_pickle(DATAFRAMESOUT + 'dfs_frame.pkl')

######## GET ENGAGEMENT FUNCTION #######
# # # get engagement from the deepface data
DATAFRAMESOUT = os.getenv("DATA_FRAMES_OUT")

dfs_frame = pd.read_pickle(DATAFRAMESOUT + 'dfs_frame.pkl')
dfs_frame = dfs_frame.iloc[1:,:] #drop first column cos empty
for df in dfs_frame['video_emotion']: # average of negative and positive valence emotions, and neutral
    df['neg_valence_avg'] = np.mean(df[['fear', 'disgust', 'angry', 'sad']], axis=1)
    df['pos_valence_avg'] = np.mean(df[['happy', 'surprise']], axis=1)
    df['neutral_avg'] = np.mean(df[['neutral']], axis=1)

    # Rolling average window length calculation:
    while int(len(df) * 0.02) > 1:
        three_percent_len = int(len(df) * 0.02)
    else:
        three_percent_len = 1
         # Taking a rolling average of these (length of the rolling average = 2% the length of the dataframe(or 2 frames
         # whichever is biggest))
    df['neg_valence_avg_roll'] = df['neg_valence_avg'].rolling(window=three_percent_len).mean()
    df['pos_valence_avg_roll'] = df['pos_valence_avg'].rolling(window=three_percent_len).mean()
    df['neutral_avg_roll'] = df['neutral_avg'].rolling(window=three_percent_len).mean()

for df in dfs_frame.groupby("person")['video_emotion']:
    print(df)

    valence_per_vid = []
    variance_per_vid = []
    total_vid_variance = []
    for emotion_df in dfs_frame['dfs']:
        valence_values = [(emotion_df['neg_valence_avg'].median()), emotion_df['pos_valence_avg'].median(),
                          emotion_df['neutral_avg'].median(), len(emotion_df)]
        # append these values to lists of lists
        valence_per_vid.append(valence_values)
        variance_per_vid.append()  # variance for each emotion in a video; Appended to a list
    # ###VALENCE###
    # # Compute valence PER VIDEO and append to frame of frames
    if dfs_frame['person_folder_names'] == i:
        dfs_frame["variance_per_vid"] = (emotion_df.iloc[:, 0:7]).var()
        dfs_frame["total_vid_pos"] = emotion_df['pos_valence_avg'].median()
        dfs_frame["total_vid_neg"] = emotion_df['neg_valence_avg'].mean()
    else: i += 1
    video_valence_df= pd.DataFrame(valence_per_vid,
                                    columns=["neg_avg_vid", "pos_avg_vid", "neutral_avg_vid", "vid_len"])
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
    # print(dfs_frame)
    # print("video stats", video_stats_df)
    # video_stats_df.to_pickle(DATAFRAMESOUT + 'cross_video_stats_df.pkl')
    # pkl_count = 0
    # for df in dfs:
    #     df.to_pickle(DATAFRAMESOUT + 'df%d' % pkl_count + 'emotion_dfs.pkl')
    #     pkl_count += 1