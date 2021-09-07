# importing dependencies
import os
import glob
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import statistics
from dotenv import load_dotenv
from scipy import stats
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

load_dotenv()


def scatter_individual(yax="pos_diff", xax="Engagement", fit="lin_reg"):
    #### Plots graphs for each individual in the dataset ###
    #   Takes as arguments:
    #                 xax, and yax -> strings with the name of the data you want to plot
    # From Engagement Calculations: "pos_mean"; "neg_mean"; "var_mean"; "pos_diff"; "neg_diff"; "var_diff"; "variance_range"; "pos_range"; "neg_range"; "percent_diff_var"; "percent_diff_pos"; "percent_diff_neg";
    # From labelled DAiSEE Dataset: "ClipID"; "Boredom"; "Engagement"; "Confusion"; "Frustration"
    #                 fit -> either "lin_reg" or "poly_fit"

    LABELS = os.getenv("LABELS")  # Import DAiSEE Labels
    labels = pd.read_csv(LABELS)  # Read into Pandas to be able to merge with rest of data

    engagement_calcs = os.getenv("ENGAGEMENT")  # Import Engagement Scores

    for i in glob.glob(engagement_calcs):
        df = pd.read_pickle(i)  # read in the dataframes iteratively

        ClipID = []
        for name in df.index:
            ClipID.append(name[1])  # get the clip ID from the index

        df["ClipID"] = ClipID  # make the clip index array into a column

        analyse = pd.merge(df, labels, how='left', on='ClipID')  # merge the df and the scores based on 'ClipID'

        xarray = []
        yarray = []

        x = analyse[xax]
        y = analyse[yax]

        #making a list of engaged = 1 (>=3) or not engaged = 0 (<=2)
        binary_y = []
        for i in y:
            if i >= 3:
                binary_y.append(0)
            else:
                binary_y.append(1)

        xarray.append(x)
        yarray.append(y)

        rsquareds = []  # empty array to collect r-squared values
        if fit == "lin_reg":
            res = stats.linregress(xarray, yarray)
            slope = [((res.slope * i) + res.intercept) for i in xarray]
            print(f"R-squared: {res.rvalue ** 2:.3f}")
            rsquareds.append(res.rvalue ** 2)
            plt.plot(xarray, yarray, 'o')
            plt.plot(xarray[0], slope[0], 'r', label='linear fit')

            plt.ylabel(yax)
            plt.xlabel(xax)
            plt.legend()
            plt.show()
        elif fit == "poly_fit":
            plt.plot(xarray, yarray, 'o')
            plt.plot(np.polyfit(xarray[0], yarray[0], deg=3), label='poly fit line')

            plt.ylabel(yax)
            plt.xlabel(xax)
            plt.legend()
            plt.show()

            plt.hist(rsquareds)
            plt.show()
    print("mean r-squared value:", statistics.mean(rsquareds))


yax="var_diff"
xax="Engagement"

LABELS = os.getenv("LABELS")  # Import DAiSEE Labels
labels = pd.read_csv(LABELS)  # Read into Pandas to be able to merge with rest of data
engagement_calcs = os.getenv("ENGAGEMENT")  # Import Engagement Scores

mean_var_eng = []
mean_var_noteng = []

appended_data = []
ClipID = []
NameID = []
for i in glob.glob(engagement_calcs):
    data = pd.read_pickle(i)  # read in the dataframes iteratively
    appended_data.append(data)
appended_data = pd.concat(appended_data)

for name in appended_data.index:
    ClipID.append(name[1])

for person in appended_data.index:
    NameID.append(person[0])

appended_data["ClipID"] = ClipID
appended_data['NameID'] = NameID

with_labels = pd.merge(appended_data, labels, how='left', on='ClipID')

binary_eng = []
for i in with_labels["Engagement"]:
    if i >= 2:
        binary_eng.append(1)
    else:
        binary_eng.append(0)

with_labels["binary_eng"] = binary_eng

# with_labels["binary_eng"].value_counts()

means = with_labels.groupby("binary_eng").mean()

person_avg = with_labels.groupby("NameID").mean()

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize = 8)
plt.rc('axes', labelsize=8)
plt.rcParams["figure.figsize"] = (3.5, 2.25)
plt.rcParams.update({'font.size': 8})

sns.countplot(x=with_labels["Engagement"])
plt.savefig(r'C:\Users\chazzers\Desktop\daisee_value_count.png',dpi=1200, bbox_inches="tight")
plt.show()
# with_labels["binary_eng"].value_counts()

# plt.scatter(person_avg["var_mean"], person_avg["Engagement"], alpha = 0.25)
# plt.xlim(0,500)
# plt.ylim(0,4)
# plt.ylabel("Engagement Score")
# plt.xlabel("Mean Variance of Emotion per Person")
# plt.savefig(r'C:\Users\chazzers\Desktop\var_eng_person_daisee.png',dpi=1200, bbox_inches="tight")
# #
# plt.scatter(person_avg["pos_mean"], person_avg["Engagement"], alpha = 0.25)
# plt.xlim(0,20)
# plt.xlabel("Average positive emotion per person (%)")
# plt.ylim(0,4)
# plt.ylabel("Engagement Score")
# plt.show()
#
# plt.scatter(person_avg["neg_mean"], person_avg["Engagement"], alpha = 0.25)
# plt.xlim(0,20)
# plt.xlabel("Average negative emotion per person (%)")
# plt.ylim(0,4)
# plt.ylabel("Engagement Score")
# plt.show()
#
# plt.scatter(with_labels["var_diff"], with_labels["Engagement"], alpha = 0.1)
# # plt.xlim(-100,100)
# plt.xlabel("Change in variance(relative to person mean)")
# plt.ylim(0,4)
# plt.ylabel("Engagement Score")
# plt.savefig(r'C:\Users\chazzers\Desktop\var_eng_pervid_daisee.png',dpi=1200, bbox_inches="tight")

#
# plt.scatter(with_labels["pos_diff"], with_labels["Engagement"], alpha = 0.015)
# # plt.xlim(-100,100)
# plt.xlabel("difference in positive emotion from the mean")
# plt.ylim(-0.5,4)
# plt.ylabel("Engagement Score")
# plt.show()
#
# plt.scatter(with_labels["neg_diff"], with_labels["Engagement"], alpha = 0.015)
# # plt.xlim(-100,100)
# plt.xlabel("difference in negative emotion from the mean")
# plt.ylim(-0.5,4)
# plt.ylabel("Engagement Score")
# plt.show()
#
# plt.scatter(with_labels["percent_diff_pos"], with_labels["Engagement"], alpha = 0.25)
# plt.xlim(-100,100)
# plt.xlabel("Normalized positive emotion")
# plt.ylim(-0.5,4)
# plt.ylabel("Engagement Score")
# plt.show()
#
# plt.scatter(with_labels["percent_diff_neg"], with_labels["Engagement"], alpha = 0.25)
# plt.xlim(-100,100)
# plt.xlabel("Normalized negative emotion")
# plt.ylim(-0.5,4)
# plt.ylabel("Engagement Score")
# plt.show()
#
# #boxplots and stripplots
# ax = sns.swarmplot(x = with_labels["Engagement"])
# # ax = sns.stripplot(x = with_labels["Engagement"], y= with_labels["percent_diff_var"], jitter = 0.2, size = 2.5 )
# plt.show()
#
# ##big plot with all pairwise combinations
# sns.set()
# sns.pairplot(with_labels)
# plt.show()
#
# binary_y = []
# for i in person_avg["Engagement"]:
#     if i >= 3:
#         binary_y.append(1)
#     else:
#         binary_y.append(0)
#     #add binary separation to dataframe
# person_avg["Binary_Engagement"] = binary_y
#     # count number of engaged and not engaged and plot them
#     # print("Engaged = 1; Not Engaged = 0:", with_labels["Binary_Engagement"].value_counts())
#     # sns.countplot(x = binary_y, data=with_labels)
#     # plt.show()
# mean_data = person_avg.groupby("Binary_Engagement").mean()
# var_eng = mean_data["var_diff"].iloc[0]
# var_noteng = mean_data["var_diff"].iloc[1]
# mean_var_eng.append(var_eng)
# mean_var_noteng.append(var_noteng)
#
# eng_var_arr = np.array(mean_var_eng)
# noteng_var_arr = np.array(mean_var_noteng)
#
# bins = np.linspace(-100, 100, 30)
# plt.hist(eng_var_arr,bins, alpha=0.5, label='emotion variance for engaged participants')
# plt.hist(noteng_var_arr,bins, alpha=0.5, label='emotion variance for unengaged participants')
# plt.legend()
# plt.show()

#     #observations from mean_data:
#     #pos_diff is higher in not engaged
#     #var_diff is higher in not engaged
#     #neg diff is lower in engaged