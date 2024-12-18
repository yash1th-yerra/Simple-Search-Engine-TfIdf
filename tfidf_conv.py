import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


# load the dataset

data = pd.read_csv('songdata.csv')


#Vectorize the text column

vectorizer = TfidfVectorizer(analyzer='word',stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['text'])


#save the vectorizer and tfidf matrix
with open('model/tfidf_vectorizer.pkl','wb') as f:
    pickle.dump(vectorizer, f)
with open('model/tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)