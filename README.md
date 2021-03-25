# Facial Emotion Recognition on Construction Sites 
*Initially attempted to make a Facial Emotion Recognition CNN based on FER2013 Kaggle Dataset. Attached model achieves around 67% accuracy.
  *Requires adding more data to training dataset to increase ability to generalize. Some tweaking with model architecture could probably improve accuracy.

*Subsequently decided to make use of DeepFace and get an 'engagement score' for each video in the aim of informing health and safety site inductions so as to make them more engaging and to ultimately improve site safety outcomes. 

*Video Training Pipeline is to prepare mp4 files for use by DeepFace

*Data Preparation and Visalization takes the data from DeepFace and calculates and plots engagment scores (cognitive and emotional)
