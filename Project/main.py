# import plotly.tools as tls
# import plotly.graph_objs as go
# import plotly.offline as py
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn import datasets, linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

chosen = ["energy", "liveness", "tempo", "valence", "loudness",
          "speechiness", "acousticness", "danceability", "instrumentalness"]

blues_df = pd.read_csv(
    "/Users/afra/Desktop/Dokument/TNM108 - Maskininlärning/Github/TNM108/Project/archive/blues_music_data.csv", )
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

data = [blues_df, rock_df, metal_df, pop_df, indie_df, hiphop_df]
specific_features = data["energy", "liveness", "tempo", "valence", "loudness",
                         "speechiness", "acousticness", "danceability", "instrumentalness"]



# text1 = data["Artist Name"] + " - " + data["Track Name"]
# text2 = text1.values

# # X = data_frame.drop(droppable, axis=1).values
# X = data[chosen].values
# y = data["danceability"].values

# min_max_scaler = MinMaxScaler()
# X = min_max_scaler.fit_transform(X)

# pca = PCA(n_components=3)
# pca.fit(X)

# X = pca.transform(X)

# py.init_notebook_mode(connected=True)

# trace = go.Scatter3d(
#     x=X[:, 0],
#     y=X[:, 1],
#     z=X[:, 2],
#     text=text2,
#     mode="markers",
#     marker=dict(
#         size=8,
#         color=y
#     )
# )

# fig = go.Figure(data=[trace])
# py.iplot(fig, filename="test-graph")
