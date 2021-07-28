# importing dependencies
import os
import glob
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import matplotlib.pyplot as plt
from scipy import stats
rng = np.random.default_rng()
#
#1) Plot variance of all videos vs all labels of all videos
df = pd.read_pickle("C:/Users/chazzers/Desktop/DAiSEE_smol/DataSet/DataFrames/cross_video_stats_df.pkl")
labels = pd.read_csv("C:/Users/chazzers/Desktop/DAiSEE_smol/DataSet/Labels/AllLabels.csv")
vid_1_eng = labels.Engagement.iloc[0:113]
vid_6_eng = labels.Engagement.iloc[282:360]
vid_7_eng = labels.Engagement.iloc[361:502]
vid_8_eng = labels.Engagement.iloc[503:511]
vid_10_eng = labels.Engagement.iloc[512:512]
vid_11_eng = labels.Engagement.iloc[513:587]
vid_12_eng = labels.Engagement.iloc[588:674]
vid_13_eng = labels.Engagement.iloc[675:682]
vid_14_eng = labels.Engagement.iloc[683:771]
vid_15_eng = labels.Engagement.iloc[772:849]

engs = pd.concat([vid_1_eng ,vid_6_eng ,vid_7_eng ,vid_8_eng ,vid_10_eng,vid_11_eng,vid_12_eng,vid_13_eng,vid_14_eng,vid_15_eng])
all_engs = engs.reset_index()
all_vids = df.iloc[0:672]
#
# # # ### Shared x-axis graph###
data1 = all_vids['neutral_avg_vid']
# # #data1 = df['percent_diff_neg']
# # #data1 = df['percent_diff_var']
data2 = all_engs
# # data2 = labels["Boredom"]
# # # data2 = labels["Confusion"]
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
x = data1
y = data2["Engagement"]

res = stats.linregress(x, y)
print(f"R-squared: {res.rvalue**2:.6f}")

plt.plot(x, y, 'o', label='original data')
plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
plt.legend()
plt.show()

