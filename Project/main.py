import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


blues_df = pd.read_csv(
    "/Users/victoriastraberg/Desktop/TNM108_Project/Project/archive/blues_music_data.csv")
# rock_df = pd.read_csv("../input/spotify-multigenre-playlists-data/rock_music_data.csv")
# metal_df = pd.read_csv("../input/spotify-multigenre-playlists-data/metal_music_data.csv")
# pop_df = pd.read_csv("../input/spotify-multigenre-playlists-data/pop_music_data.csv")
# indie_df = pd.read_csv("../input/spotify-multigenre-playlists-data/indie_alt_music_data.csv")
# alt_df = pd.read_csv("../input/spotify-multigenre-playlists-data/alternative_music_data.csv")
# hiphop_df = pd.read_csv("../input/spotify-multigenre-playlists-data/hiphop_music_data.csv")

print(blues_df.head())


