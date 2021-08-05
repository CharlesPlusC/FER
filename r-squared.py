# importing dependencies
import os
import glob
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import statistics
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from scipy import stats
load_dotenv()

labels = pd.read_csv("D:/DAiSEE/DAiSEE/DataSet/Labels/AllLabels.csv") #Import Engagement Labels

frame_count = 0 # start a frame counter


df = pd.read_pickle("D:/DAiSEE/DAiSEE/DataSet/DataFrames/110001engagement_calcs.pkl")
df_name = df.index[0][0]

ClipID = []
for name in df.index:
    ClipID.append(name[1])

df["ClipID"] = ClipID

analyse = pd.merge(df, labels, how='left', on='ClipID')

# ## linear regression graph###
x = analyse['percent_diff_var']
y = analyse['Engagement']

res = stats.linregress(x, y)
print(f"R-squared: {res.rvalue**2:.6f}")

plt.plot(x, y, 'o', label='original data')
plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
plt.legend()
plt.show()