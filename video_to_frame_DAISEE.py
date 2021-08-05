import os
import glob
import cv2
import numpy as np
import pandas as pd
import statistics
from deepface import DeepFace
from dotenv import load_dotenv
load_dotenv()

#TODO: Explain what the required file is
#TODO: check for the required file structure. check if populated? Or we could just explain this but might make it easier to have a coded check.
#TODO: is there any way of specifying the the paths of the files (like in .env) through the UI?

# split each video into component frames at a determined frame rate
def split_vid(required_frame_rate, video_frame_rate):
    PATHIN = os.getenv("PATH_IN")
    PATHOUT = os.getenv("PATH_OUT")

    video_paths = []  # list of videos to get frames from
    video_names = []  # list of only the video names

    for folder in os.listdir(PATHIN):
        folder = PATHIN + "/" + folder
        for vid in os.listdir(folder):
            vid = folder + "/" + vid
            for video_name in os.listdir(vid):
                video_names.append(video_name)
                video_name = vid + "/" + video_name
                video_paths.append(video_name)
        vid_count = 0

    for i in video_paths:
        cap = cv2.VideoCapture(i)
        vid_count += 1
        success = True
        frame_count = 1  # reset frame count to 1 at the start of every new video
        while success:
            success, image = cap.read()
            print('read a new frame:', success)
            if frame_count % (video_frame_rate * required_frame_rate) == 0:
                cv2.imwrite(PATHOUT + 'video%d' % vid_count + 'frame%d.jpg' % frame_count, image)
                print("frame written")
            frame_count += 1

# takes the frames from one video at a time, puts them into an array and passes them through deepface
def get_emotion():
    DATAFRAMESOUT = os.getenv("DATA_FRAMES_OUT")
    PATHIN = os.getenv("PATH_IN")
    PATHOUT = os.getenv("PATH_OUT")
    img_array = []
    dfs = []
    video_paths = []
    video_names = []
    person_folder_names = []

    # added the video counter here so it does not depend on the previous function
    video_counter = 0  # how many videos there are
    for i in os.listdir(PATHIN):
        person_folder = PATHIN + i
        for vid_folder in os.listdir(person_folder):
            vid_folder = person_folder + "/" + vid_folder
            for video in os.listdir(vid_folder):
                video_names.append(video)  # saving just the name of the video into an array
                video = vid_folder + "/" + video
                video_paths.append(video)  # saving the whole path of the video into another array
                person_folder_names.append(i)  # add an instance of 'folder name' for each video
                video_counter += 1
    for i in range(0, video_counter, 1):
        for filename in glob.glob(PATHOUT + 'video%d' % i + 'frame*.jpg'):
            # Read in the relevant images
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        # Pass them through deepface
        face_FER = DeepFace.analyze(img_path=img_array, actions=['emotion'], enforce_detection=False)
        img_array = []  # reset image array to be blank
        data = face_FER
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
        dfs.append(df)  # TODO: need to index this with 'video names'
    #
    index_arrays = (person_folder_names,video_names)
    dfs_index = pd.MultiIndex.from_arrays(index_arrays, names=["person", "video"])
    dfs_frame = pd.DataFrame(dfs,index = dfs_index, columns=['video_emotion']) #dataframe indexed by folder name and video name (two levels)
    dfs_frame.to_pickle(DATAFRAMESOUT + 'dfs_frame.pkl')

# get engagement from the deepface data
def get_engagement():
    DATAFRAMESOUT = os.getenv("DATA_FRAMES_OUT")

    dfs_frame = pd.read_pickle(DATAFRAMESOUT + 'dfs_frame.pkl')
    dfs_frame = dfs_frame.iloc[1:, :]  # drop first column cos empty?
    # grouping the emotions in each video
    for df in dfs_frame['video_emotion']:  # average of negative and positive valence emotions, and neutral PER FRAME
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

        # making a new column with the average per video
        df['neutral_person_avg'] = df['neutral_avg'].median()
        df["variance_per_vid"] = (df.iloc[:, 0:7].var()).mean()
        df["total_vid_pos"] = df['pos_valence_avg'].median()
        df["total_vid_neg"] = df['neg_valence_avg'].mean()

    # # # now calculating emotions 'per person'
    for person_name, data in dfs_frame.groupby(level=0):
        positive_array = []
        negative_array = []
        variance_array = []

        pos_diff_array = []
        neg_diff_array = []
        var_diff_array = []

        for i in range(0, len(data.index), 1):
            positive_array.append(data['video_emotion'][i]["total_vid_pos"].mean())
            pos_mean = statistics.mean(positive_array)

            negative_array.append(data['video_emotion'][i]["total_vid_neg"].mean())
            neg_mean = statistics.mean(negative_array)

            variance_array.append(data['video_emotion'][i]["variance_per_vid"].mean())
            var_mean = statistics.mean(variance_array)

            # Mean of all the pos,neg,var values in one folder, appended as a column to the frame of frames
            data["pos_mean"] = pos_mean
            data["neg_mean"] = neg_mean
            data["var_mean"] = var_mean

            # subtract the single values(pos_mean,neg_mean, var_mean) from the video emotions
        for i in range(0, len(data['video_emotion'].index), 1):
            pos_diff = (data["pos_mean"][i]) - (data['video_emotion'][i]["total_vid_pos"].mean())
            pos_diff_array.append(pos_diff)

            neg_diff = (data["neg_mean"][i]) - (data['video_emotion'][i]["total_vid_neg"].mean())
            neg_diff_array.append(neg_diff)

            var_diff = (data["var_mean"][i]) - (data['video_emotion'][i]["variance_per_vid"].mean())
            var_diff_array.append(var_diff)

        data['pos_diff'] = pos_diff_array
        data['neg_diff'] = neg_diff_array
        data['var_diff'] = var_diff_array

        # Getting the differences between the highest and lowest values for emotions and variance
        data["variance_range"] = data["var_diff"].max() - data["var_diff"].min()
        data["pos_range"] = data["pos_diff"].max() - data["pos_diff"].min()
        data["neg_range"] = data["neg_diff"].max() - data["neg_diff"].min()

        # Using the above ranges to calculate the percentage difference
        data["percent_diff_var"] = (data["var_diff"] / data["variance_range"]) * 100
        data["percent_diff_pos"] = (data["pos_diff"] / data["pos_range"]) * 100
        data["percent_diff_neg"] = (data["neg_diff"] / data["neg_range"]) * 100

        print("engagement calcs for %s:" % person_name, data)
        data.to_pickle(DATAFRAMESOUT + '%s' % person_name + 'engagement_calcs.pkl')
#
# if __name__ == "__main__":
#     video_counter = split_vid(2, 30)
#     dfs = get_emotion()
#     get_engagement(dfs)
