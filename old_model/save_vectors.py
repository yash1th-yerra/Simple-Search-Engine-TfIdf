# save_vectors.py

import pandas as pd
import numpy as np
import gensim
from nltk.tokenize import word_tokenize

# Load the dataset
data = pd.read_csv('../songdata.csv')

# Combine 'song', 'artist', and 'text' into a single string, lowercased
data['tags'] = (data['song'].str.lower() + " " +
                data['artist'].str.lower() + " " +
                data['text'].str.lower())

# Load the pre-trained Word2Vec model
model = gensim.models.Word2Vec.load("word2vec_model.model")

# Function to store song vectors in a .npy file
def store_song_vectors(data, filename="song_vectors.npy"):
    song_vectors = []
    for song in data['tags']:
        # Tokenize and create the vector for the song
        song_tokens = word_tokenize(song.lower())
        song_vector = np.mean([model.wv[token] for token in song_tokens if token in model.wv], axis=0)
        song_vectors.append(song_vector)

    # Save the vectors as a NumPy array
    np.save(filename, np.array(song_vectors))
    print(f"Song vectors saved to {filename}")

if __name__ == "__main__":
    store_song_vectors(data)  # Run the function to save vectors
