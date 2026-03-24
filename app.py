import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv('movies.csv')

# Ensure necessary columns exist
for col in ['description', 'genres', 'poster_path']:
    if col not in movies.columns:
        movies[col] = ''
    movies[col] = movies[col].fillna('')

# Combine features
movies['combined'] = movies['description'] + " " + movies['genres']

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined'])

# Cosine similarity
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

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="wide")

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
                poster_url = (
                    f"https://image.tmdb.org/t/p/w500{row['poster_path']}"
                    if row['poster_path']
                    else "https://via.placeholder.com/150x220.png?text=No+Poster"
                )
                st.image(poster_url, width=150)
                st.write(f"**{row['title']}**")
                st.write(f"🎭 {row['genres']}")
