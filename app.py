import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔑 Add your TMDB API key here
API_KEY = "YOUR_API_KEY_HERE"

# Load dataset
movies = pd.read_csv('movies.csv')

# Fill missing values
for col in ['title', 'description', 'genres']:
    if col not in movies.columns:
        movies[col] = ''
    movies[col] = movies[col].fillna('')

# Combine features
movies['combined'] = movies['description'] + " " + movies['genres']

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])

# Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def get_recommendations(title):
    try:
        idx = movies[movies['title'].str.lower() == title.lower()].index[0]
    except IndexError:
        return pd.DataFrame()

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]

    return movies.iloc[movie_indices]

# 🔥 Fetch poster from TMDB
def fetch_poster(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
    data = requests.get(url).json()

    try:
        poster_path = data['results'][0]['poster_path']
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return "https://via.placeholder.com/150x220.png?text=No+Poster"

# UI
st.title("🎬 Movie Recommendation System")

movie_title = st.selectbox(
    "Select a movie you like:",
    movies['title'].dropna().unique()
)

if st.button("Show Recommendations"):
    recommendations = get_recommendations(movie_title)

    if recommendations.empty:
        st.warning("No recommendations found.")
    else:
        cols = st.columns(len(recommendations))

        for i, (_, row) in enumerate(recommendations.iterrows()):
            with cols[i]:
                poster = fetch_poster(row['title'])
                st.image(poster, width=150)
                st.write(f"**{row['title']}**")
                st.write(f"🎭 {row['genres']}")
