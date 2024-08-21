import pandas as pd
from qdrant_client import QdrantClient, models
from collections import defaultdict

# Load the data
tags = pd.read_csv("./data/ml-latest-small/tags.csv")
movies = pd.read_csv("./data/ml-latest-small/movies.csv")
ratings = pd.read_csv("./data/ml-latest-small/ratings.csv")

# Initialize Qdrant client and create collections
def init_qdrant():
    qdrant = QdrantClient(":memory:")  # Use in-memory for simplicity
    qdrant.create_collection(
        "movielens", vectors_config={}, sparse_vectors_config={"ratings": models.SparseVectorParams()}
    )
    return qdrant

# Load data and upload to Qdrant
def load_data(qdrant):
    ratings['normalized_rating'] = (ratings.rating - ratings.rating.mean(axis=0)) / ratings.rating.std()

    user_sparse_vectors = defaultdict(lambda: {"values": [], "indices": []})
    for row in ratings.itertuples():
        user_sparse_vectors[row.userId]["values"].append(row.normalized_rating)
        user_sparse_vectors[row.userId]["indices"].append(row.movieId)

    def data_generator():
        for user_id, vector in user_sparse_vectors.items():
            yield models.PointStruct(
                id=user_id, vector={"ratings": vector}, payload={}
            )

    qdrant.upload_points("movielens", data_generator())

# Function to input and normalize ratings
def input_ratings(user_ratings, ratings):
    final_ratings = {}
    
    mean_rating = ratings.rating.mean()
    std_rating = ratings.rating.std()

    for movie_id, user_rating in user_ratings.values():
        normalized_input_rating = (user_rating - mean_rating) / std_rating
        final_ratings[movie_id] = normalized_input_rating
    
    return final_ratings

# Search and recommendation function
def recommend_movies(qdrant, movies, my_ratings):
    def to_vector(ratings):
        vector = models.SparseVector(values=[], indices=[])
        for movieId, rating in ratings.items():
            vector.values.append(rating)
            vector.indices.append(movieId)
        return vector

    user_vector = to_vector(my_ratings)

    results = qdrant.search(
        "movielens",
        query_vector=models.NamedSparseVector(name="ratings", vector=user_vector),
        with_vectors=True,
        limit=20,
    )

    movie_scores = defaultdict(lambda: 0)
    for user in results:
        user_scores = user.vector["ratings"]
        for idx, rating in zip(user_scores.indices, user_scores.values):
            if idx in my_ratings:
                continue
            movie_scores[idx] += rating

    top_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
    recommended_movies = [movies[movies.movieId == movieId].title.values[0] for movieId, score in top_movies[:5]]
    
    return recommended_movies
