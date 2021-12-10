########################################################################################
# Song recommender
# Authors: Afra Farkhooy, Victoria Stråberg
# Project in course TNM108
########################################################################################

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

raw_data_with_measures = pd.read_csv(
    "/Users/afra/Desktop/Dokument/TNM108 - Maskininlärning/Projekt/SpotifyFeatures.csv")

#print(raw_data_with_measures.head())

################################################### DATA PRE-PROCESSING ###################################################

#Dropping unnecessary columns
raw_data_with_measures = raw_data_with_measures.drop(["genre", "popularity", "duration_ms", "key", "time_signature"], axis="columns")

#Dropping duplicate values
raw_data_with_measures = raw_data_with_measures.drop_duplicates(subset=['track_id'])

#Checking missing values
def display_missing(data):
    for col in data.columns.tolist():
        print('{} column missing values: {}'.format(
            col, raw_data_with_measures[col].isnull().sum()))
    print('\n')
display_missing(raw_data_with_measures)

#Scaling the data
song_data = raw_data_with_measures.loc[:, [
    'acousticness', 'danceability', 'energy', 'liveness', 'speechiness', 'valence']]

scaler = MinMaxScaler()
song_features = pd.DataFrame()

for col in song_data.iloc[:, :].columns:
    scaler.fit(song_data[[col]])
    song_features[col] = scaler.transform(song_data[col].values.reshape(-1, 1)).ravel()

#Merging the new scaled data to the original datafile and dropping the unscaled data. final_merge_df is our new datafile.
data_to_merge = raw_data_with_measures.drop(['acousticness', 'danceability', 'energy', 'liveness', 'speechiness', 'valence'], axis="columns")
final_merge_df = data_to_merge.join(song_features).dropna()

# print(df.loc[1152, :])
# print(df.loc[770, :])

################################################### COSINE SIMILARITY ###################################################

def rank_song_similarity_by_measure(data, song, artist):

    data.rename(columns={'track_name': f'Song', 'artist_name': f'Artist'}, inplace=True)

    #Picking out the artist and the song
    artist = data.loc[data['Artist'] == artist]
    song = artist.loc[artist['Song'] == song]

    #Creating new data including only the chosen song attributes 
    song_and_artist_data = song.drop(["Song", "Artist", "track_id", "instrumentalness", "loudness", "tempo", "mode"], axis="columns")

    #Copy of input data
    similar_data = data.copy()

    #Picking out the chosen song attributes from the input data
    data_values = similar_data.loc[:, ['acousticness', 'danceability', 'energy', 'liveness', 'speechiness', 'valence']]

    #Cosine similarity
    cos_sim = cosine_similarity(song_and_artist_data, data_values)

    #Printing out the similar songs

    for i in range(int(5)):
        
        most_similar = np.argmax(cos_sim[0], axis=0)
        # print(data.iloc[[most_similar], [0,1]]) #0 column artist, 1 column for song 
        print(data.iloc[most_similar, [0,1]]) #0 column artist, 1 column for song 
        print('Cosine Similarity:', (cos_sim[0][most_similar]))
        print("") 
        cos_sim[0][most_similar] = 0

rank_song_similarity_by_measure(final_merge_df, "Material Girl", "Madonna")

