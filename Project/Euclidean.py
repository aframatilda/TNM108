from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd

df = pd.read_csv(
    "/Users/afra/Desktop/Dokument/TNM108 - Maskininlärning/Projekt/SpotifyFeatures.csv")
print(df.columns)

######## DATA PRE-PROCESSING ########

# Dropping unnecessary features
df = df.drop(["genre", "popularity",
             "duration_ms", "key", "mode", "time_signature"], axis="columns")
# print(df.columns)

# dropping duplicate values
df = df.drop_duplicates(subset=['track_id'])

# print(df.loc[1152, :])
# print(df.loc[770, :])

# Checking missing values


def display_missing(df):
    for col in df.columns.tolist():
        print('{} column missing values: {}'.format(
            col, df[col].isnull().sum()))
    print('\n')

# display_missing(df)

######## COSINE SIMILARITY ########


def rank_song_similarity_by_measure(data, song, artist):

    song_and_artist_data = data[(data['artist_name'] == artist) & (
        data['track_name'] == song)]

    similarity_data = data.copy()

    data_values = similarity_data.loc[:, [
        'acousticness', 'danceability', 'energy', 'liveness', 'speechiness', 'valence']]

    similarity_data['Similarity euclidean'] = euclidean_distances(
        data_values, data_values.to_numpy()[song_and_artist_data.index[0], None]).squeeze()

    similarity_data.rename(
        columns={'track_name': f'Song', 'artist_name': f'Artist'}, inplace=True)

    similar_song = similarity_data.sort_values(
        by='Similarity euclidean', ascending=False)

    print("\n")
    print(f'Songs Similar to {song}')
    print("\n")
    similar_song = similar_song[['Artist', 'Song', 'Similarity euclidean']]

    similar_song.reset_index(drop=True, inplace=True)

    return similar_song.iloc[1:5]


print(rank_song_similarity_by_measure(
    df, "All I Do Is Win", "DJ Khaled"))
