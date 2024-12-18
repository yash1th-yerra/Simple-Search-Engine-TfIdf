from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset and models
data = pd.read_csv('songdata.csv')
with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('model/tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    if not query:
        return render_template('results.html', query=query, results=[])

    # Transform query to TF-IDF vector
    query_vec = vectorizer.transform([query])

    # Compute cosine similarity
    similarities = cosine_similarity(tfidf_matrix, query_vec).reshape(-1)
    top_indices = similarities.argsort()[-10:][::-1]  # Top 10 results

    results = [
        {
            'title': data.iloc[i]['song'],  # Adjust column names
            'artist': data.iloc[i]['artist'],
            'score': round(similarities[i], 4)
        }
        for i in top_indices
    ]

    return render_template('results.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)
