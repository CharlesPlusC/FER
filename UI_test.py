import os
from video_to_frame_DAISEE import get_emotion, get_engagement, split_vid
from dotenv import load_dotenv

load_dotenv()

from tkinter import *

root = Tk()
root.title('Engagement analysis')
root.iconbitmap('C:/Users/chazzers/Desktop/testicon.ico')

#add an entry that collects two integers to pass to Emo function.
#
e = Entry(root,width=35,borderwidth=5)
e.grid(row=0, column=0, columnspan=2, padx=10, pady=10)


def Split_Emo_Eng():
    pass
    # split_vid()
    # dfs = get_emotion()
    # get_engagement(dfs)


def Split():
    pass
    # split_vid()


def Emo():
    video_counter = split_vid(2,30)
    # get_emotion()


def Eng():
    pass
    # get_engagement(dfs)

# Functions Button
do_all_button = Button(root, text="split+emotion+engagement", padx=40, pady=40, command=Split_Emo_Eng)
split_button = Button(root, text="split", padx=40, pady=40, command=Split)
emo_button = Button(root, text="emotion", padx=40, pady=40, command=Emo)
eng_button = Button(root, text="engagement", padx=40, pady=40, command=Eng)

do_all_button.grid(row=1, column=0)
split_button.grid(row=1, column=1)
emo_button.grid(row=2, column=0)
eng_button.grid(row=2, column=1)

# # Exit button
# button_quit = Button(root, text="Exit program", command=root.quit)
# button_quit.pack()

root.mainloop()
