from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd

df = pd.read_csv(
    "/Users/afra/Desktop/Dokument/TNM108 - Maskininlärning/Projekt/data.csv")
# Features

print(df.loc[1394, :])
print(df.loc[1188, :])

# print(df.columns)

######## DATA PRE-PROCESSING ########

# Dropping unnecessary features
# df = df.drop(["Unnamed: 0", "duration_ms", "key",
#              "mode" "time_signature", "target"])
# print(df.columns)

# # Checking missing values


# def display_missing(df):
#     for col in df.columns.tolist():
#         print('{} column missing values: {}'.format(
#             col, df[col].isnull().sum()))
#     print('\n')


# display_missing(df)

######## COSINE SIMILARITY ########


def rank_song_similarity_by_measure(data, song, artist):

    song_and_artist_data = data[(data['artist'] == artist) & (
        data['song_title'] == song)]

    similarity_data = data.copy()

    data_values = similarity_data.loc[:, [
        'acousticness', 'danceability', 'energy', 'liveness', 'speechiness', 'valence']]

    similarity_data['Similarity'] = cosine_similarity(
        data_values, data_values.to_numpy()[song_and_artist_data.index[0], None]).squeeze()

    similarity_data.rename(
        columns={'song_title': f'Song', 'artist': f'Artist'}, inplace=True)

    similar_song = similarity_data.sort_values(
        by='Similarity', ascending=False)

    print(f'Songs Similar to {song}')
    similar_song = similar_song[['Artist', 'Song', 'Similarity']]

    similar_song.reset_index(drop=True, inplace=True)

    return similar_song.iloc[1:5]


print(rank_song_similarity_by_measure(
    df, "Turn Down for What", "DJ Snake"))
