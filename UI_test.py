from video_to_frame_DAISEE import get_emotion, get_engagement, split_vid
from dotenv import load_dotenv
load_dotenv()

from tkinter import *
root = Tk()

def printName():
    print("I smell")

button_1 = Button(root, text = "function1", command = get_emotion)
button_1.pack()
root.mainloop()