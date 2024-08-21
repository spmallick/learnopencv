import streamlit as st
import pandas as pd
from main import init_qdrant, load_data, input_ratings, recommend_movies

# Initialize Qdrant and load data
qdrant = init_qdrant()
load_data(qdrant)

# Load movies
movies = pd.read_csv("./data/ml-latest-small/movies.csv")
ratings = pd.read_csv("./data/ml-latest-small/ratings.csv")

# Initialize session state to store user ratings
if "user_ratings" not in st.session_state:
    st.session_state["user_ratings"] = {}

# Streamlit app interface
st.title("Movie Recommendation System")

# Movie selection and rating
movie_titles = movies["title"].tolist()

# Movie search and selection using multiselect
selected_movies = st.multiselect("Search and select movies to rate", movie_titles)

if selected_movies:
    for movie in selected_movies:
        rating = st.slider(f"Rate {movie}", 0.0, 5.0, 0.0, 0.5)
        if st.button(f"Add {movie}"):
            movie_id = movies[movies.title == movie].movieId.iloc[0]
            st.session_state["user_ratings"][movie] = (movie_id, rating)
            st.write(f"Added: {movie} with a rating of {rating}")
else:
    st.write("Select movies to rate from the dropdown.")

# Clear button to reset all inputs
if st.button("Clear Selections"):
    st.session_state["user_ratings"] = {}
    st._set_query_params()  # Reset the app state
    
# Display current ratings
if st.session_state["user_ratings"]:
    st.write("Current Movie Ratings:")
    for movie, (movie_id, rating) in st.session_state["user_ratings"].items():
        st.write(f"{movie}: {rating}")

# Get recommendations
if st.button("Get Recommendations"):
    if st.session_state["user_ratings"]:
        final_ratings = input_ratings(st.session_state["user_ratings"], ratings)
        recommendations = recommend_movies(qdrant, movies, final_ratings)

        if recommendations:
            st.header("Recommended Movies for You")
            for movie in recommendations:
                st.write(movie)
        else:
            st.info("No recommendations found based on your ratings.")
    else:
        st.warning("Please rate at least one movie to get recommendations.")
