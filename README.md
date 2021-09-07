# Tracking engagement through the use of facial affect #
- Initially attempted to make a Facial Emotion Recognition CNN based on FER2013 Kaggle Dataset. Attached model achieves around 67% accuracy. Can be found in *CNN_v2.ipynb*

- Subsequently decided to make use of DeepFace and get an 'engagement score' for each video in the aim of informing health and safety site inductions so as to make them more engaging and to ultimately improve site safety outcomes. 

- Jupyter notebooks used as early rough drafts of the program (video preocessing, dataframe extraction from deepface, engagement calculation and comparison with observations)

### Engagement calculator program ###
#### Disclaimer: ####
The following code provides a first attempt at calculating 'engagament' within a given video/set of videos. There are many issues with this method including the fact that facial emotion (a.k.a affect) is only representative of **external state** and so can only be used to infer external markers of engagement. This method makes no claims of being able to identify any more than this.

#### Using the code ####
Required file structure to use main.py/ video_to_frame_DAISEE.py: ![file structure](https://user-images.githubusercontent.com/66725307/127992724-2c24bc24-f5fe-4088-83a9-f7c3c4e583cc.jpeg)

The main functions are as follows:
- **split_vid()**: Simply splits the video into its frames. Takes as arugments the frame rate of the camera used and the number of frames per second required.
- **get_emotion()**: applies DeepFace (Serengil, 2020) to split frames and outputs dataframes
- **get_engagement()**: processing of the data from get_emotion() to return 'engagement' of a participant relative to themselves in any one clip

*UI.py* contains a basic user interface.
