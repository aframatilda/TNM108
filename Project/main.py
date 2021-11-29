import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.target import JointPlotVisualizer

blues_df = pd.read_csv(
    "/Users/afra/Desktop/Dokument/TNM108 - Maskininlärning/Github/TNM108/Project/archive/blues_music_data.csv")
rock_df = pd.read_csv(
    "/Users/afra/Desktop/Dokument/TNM108 - Maskininlärning/Github/TNM108/Project/archive/rock_music_data.csv")
metal_df = pd.read_csv(
    "/Users/afra/Desktop/Dokument/TNM108 - Maskininlärning/Github/TNM108/Project/archive/metal_music_data.csv")
pop_df = pd.read_csv(
    "/Users/afra/Desktop/Dokument/TNM108 - Maskininlärning/Github/TNM108/Project/archive/pop_music_data.csv")
indie_df = pd.read_csv(
    "/Users/afra/Desktop/Dokument/TNM108 - Maskininlärning/Github/TNM108/Project/archive/indie_alt_music_data.csv")
# alt_df = pd.read_csv(
#     "./Users/afra/Desktop/Dokument/TNM108 - Maskininlärning/Github/TNM108/Project/archive/alternative_music_data.csv", usecols=col_list)
hiphop_df = pd.read_csv(
    "/Users/afra/Desktop/Dokument/TNM108 - Maskininlärning/Github/TNM108/Project/archive/hiphop_music_data.csv")

all_genres = [blues_df, rock_df, metal_df, pop_df, indie_df, hiphop_df]
feature_names = ["acousticness", "danceability", "energy",
                 "instrumental", "liveness", "loudness", "tempo", "valence"]

X, y = all_genres[feature_names], all_genres["genre"]

features = np.array(feature_names)

visualizer = FeatureCorrelation(labels=features)

plt.rcParams["figure.figsize"] = (20, 20)
visualizer.fit(X, y)
visualizer.show()
