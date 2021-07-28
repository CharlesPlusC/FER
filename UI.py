import os
from video_to_frame_DAISEE import get_emotion, get_engagement, split_vid
from dotenv import load_dotenv

load_dotenv()

from tkinter import *

root = Tk()
root.title('Engagement analysis')
# root.iconbitmap('C:/Users/chazzers/Desktop/testicon.ico')

# Collects two integers to pass to Emo function(video_frame_rate, required_frame_rate).
# make the text for the entry
video_frame_rate_collect = Entry(root, width=35, borderwidth=5)
video_frame_rate_collect.grid(row=6, column=1, padx=10, pady=10)

# make the button for it
video_frame_rate_collect_label = Label(root, text='Frame rate of the camera used:')
video_frame_rate_collect_label.grid(row=6, column=0)

# make the entry box
required_frame_rate_collect = Entry(root, width=35, borderwidth=5)
required_frame_rate_collect.grid(row=7, column=1, padx=10, pady=10)

# make the button for it
required_frame_rate_collect_label = Label(root, text='Desired processing rate of frames:')
required_frame_rate_collect_label.grid(row=7, column=0)


# collect the arguments
def get_split_args():
    global required_frame_rate, video_frame_rate
    required_frame_rate = float(required_frame_rate_collect.get())
    video_frame_rate = int(video_frame_rate_collect.get())
    print("Required Frame Rate: %s\nvideo_frame_rate: %s" % (required_frame_rate, video_frame_rate))


collect_args_button = Button(root, text='Apply Frame Rate Settings', command=get_split_args).grid(row=9, column=1,
                                                                                                  pady=4)

# Explain what the buttons do
Do_All_Explainer = Label(root,
                         text='Use the "Do All" button to split the videos, analyse them for emotion, and return a '
                              'dataframe with enagement scores')

Split_Explainer = Label(root, text='Use the "Split" button to split the videos into their component frames')
# TODO: add something about specifying frame rate and desired rate

Emotion_Explainer = Label(root, text='Use the "Emotion" button to analyse the videos for emotion using DeepFace and'
                                     ' return dataframes containing emotion for each video frame')  # TODO: store the outputs of DeepFace in a folder

Engagement_Explainer = Label(root, text='Use the "Engagement" button to generate engagement scores from the emotion'
                                        ' dataframes. The outputs will be stored as dataframes in the "DataFrames" folder')


def Split_Emo_Eng():
    split_vid(1, 30)
    dfs = get_emotion()
    get_engagement(dfs)


def Split():
    split_vid(required_frame_rate, video_frame_rate)


def Emotion():
    get_emotion()


def Engagement():
    get_engagement(dfs)


# Functions Button
do_all_button = Button(root, text="Do All", padx=40, pady=40, command=Split_Emo_Eng)
split_button = Button(root, text="1.Split", padx=40, pady=40, command=Split)
emotion_button = Button(root, text="2.Emotion", padx=40, pady=40, command=Emotion)
engagement_button = Button(root, text="3.Engagement", padx=40, pady=40, command=Engagement)

do_all_button.grid(row=4, column=0)
split_button.grid(row=4, column=1)
emotion_button.grid(row=5, column=0)
engagement_button.grid(row=5, column=1)

# Explanation text
Do_All_Explainer.grid(row=0, column=0, columnspan=2)
Split_Explainer.grid(row=1, column=0, columnspan=2)
Emotion_Explainer.grid(row=2, column=0, columnspan=2)
Engagement_Explainer.grid(row=3, column=0, columnspan=2)

# # Exit button
# button_quit = Button(root, text="Exit program", command=root.quit)
# button_quit.grid(row=8, column=0, columnspan=2)

root.mainloop()
