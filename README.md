🎵 Simple Search Engine for Songs

A Flask-based Search Engine that allows users to search for songs using lyrics snippets! This project demonstrates how to implement a basic text search functionality with TF-IDF Vectorization and Cosine Similarity for ranking results.

🚀 Features

    🔍 Search Functionality: Enter a phrase or a snippet of lyrics to find matching songs.
    📊 TF-IDF Vectorization: Efficient text representation for accurate matching.
    🧠 Cosine Similarity: Ranks search results based on relevance.
    🎨 User-Friendly Interface: A simple and responsive web design built with Flask and CSS.
    🌐 Deployable: Ready for deployment on platforms like Render, Railway, or Heroku.

🛠️ Technologies Used

    - Backend: Flask (Python)
    - Text Processing: Scikit-learn (TF-IDF and Cosine Similarity)
    - Frontend: HTML5, CSS3
    - Deployment: Render 

📝 How It Works

    - The dataset (songdata.csv) is loaded and preprocessed.
    - Lyrics are transformed into vectors using TF-IDF.
    - User queries are vectorized and compared with the dataset using Cosine Similarity.
    - Results are ranked by relevance and displayed to the user.

📂 Project Structure

.
├── app.py               # Main Flask application
├── templates/
│   ├── layout.html      # Base HTML layout
│   ├── search.html      # Search page template
├── static/
│   ├── style.css        # CSS for styling
├── songdata.csv         # Dataset of songs and lyrics
├── requirements.txt     # Python dependencies
└── Procfile             # Deployment configuration (for Render/Heroku)

🚀 Getting Started
1️⃣ Clone the Repository

git clone https://github.com/your-username/song-search-engine.git
cd song-search-engine

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Application

python app.py

Visit http://127.0.0.1:5000 in your browser to start searching!

📖 Future Enhancements

   - Add autocomplete for the search bar.
   - Integrate a database for faster indexing and querying.
   - Implement advanced ranking algorithms for more accurate results.
   - Enhance the UI with more modern styles and animations.

- Live Demo : https://simple-search-engine-tfidf.onrender.com/

🤝 Contributions

Contributions, issues, and feature requests are welcome! Feel free to fork this repository and submit a pull request.
🛡️ License

This project is licensed under the MIT License.
