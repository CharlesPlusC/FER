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

        xarray.append(x)
        yarray.append(y)

        if fit == "lin_reg":
            res = stats.linregress(xarray, yarray)
            slope = [((res.slope * i)+res.intercept) for i in xarray]
            print(f"R-squared: {res.rvalue ** 2:.3f}")
            plt.plot(xarray, yarray, 'o', label='original data')
            plt.plot(xarray[0], slope[0], 'r', label='fitted line')
            plt.show()
        elif fit == "poly_fit":
            plt.plot(xarray, yarray, 'o', label='original data')
            plt.plot(np.polyfit(xarray[0], yarray[0], deg = 3))
            plt.show()