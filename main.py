from video_to_frame_DAISEE import get_emotion, get_engagement, split_vid
from dotenv import load_dotenv
load_dotenv()

video_counter = split_vid(2,30)
dfs =get_emotion()
get_engagement(dfs)
