# app.py

import streamlit as st
import pandas as pd
import numpy as np
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

# Load the dataset
data = pd.read_csv('../songdata.csv')

# Combine 'song', 'artist', and 'text' into a single string, lowercased
data['tags'] = (data['song'].str.lower() + " " +
                data['artist'].str.lower() + " " +
                data['text'].str.lower())

# Load the pre-trained Word2Vec model
model = gensim.models.Word2Vec.load("word2vec_model.model")

# Function to load precomputed song vectors
def load_song_vectors(filename="song_vectors.npy"):
    return np.load(filename)

# Load the precomputed vectors for fast search
song_vectors = load_song_vectors()

# Function to search for the most similar songs
def search_similar_songs(query, top_n=10):
    # Tokenize the query and create its vector
    query_tokens = word_tokenize(query.lower())  # Lowercased query and tokenize
    query_vector = np.mean([model.wv[token] for token in query_tokens if token in model.wv], axis=0)

    if query_vector is None:
        return []

    # Compute cosine similarity between the query vector and precomputed song vectors
    similarities = cosine_similarity([query_vector], song_vectors).flatten()

    # Get the top N most similar songs
    top_indices = similarities.argsort()[-top_n:][::-1]
    result = []
    for idx in top_indices:
        song = data.iloc[idx]
        result.append({
            "song": song['song'],
            "artist": song['artist'],
            "similarity": similarities[idx]
        })

    return result

# Streamlit UI
def main():
    st.title("Song Search Engine")

    query = st.text_input("Enter a song name or description to search:")
    if query:
        st.write("Searching for similar songs...")

        # Search similar songs based on the query
        similar_songs = search_similar_songs(query)

        if similar_songs:
            st.write(f"Found {len(similar_songs)} similar songs:")

            # Display the results in a table format
            for song in similar_songs:
                st.write(
                    f"**Song:** {song['song']} | **Artist:** {song['artist']} | **Similarity:** {song['similarity']:.4f}")
        else:
            st.write("No similar songs found.")


if __name__ == "__main__":
    main()
