import pandas as pd
from qdrant_client import QdrantClient, models
from collections import defaultdict


# Initialize Qdrant client and create collections
def init_qdrant():
    qdrant = QdrantClient(":memory:")  # Use in-memory for simplicity
    qdrant.create_collection(
        "movielens", vectors_config={}, sparse_vectors_config={"ratings": models.SparseVectorParams()}
    )
    return qdrant


# Load data and upload to Qdrant
def load_data(qdrant):
    tags = pd.read_csv("./data/ml-latest-small/tags.csv")
    movies = pd.read_csv("./data/ml-latest-small/movies.csv")
    ratings = pd.read_csv("./data/ml-latest-small/ratings.csv")

    # Normalize ratings
    ratings.rating = (ratings.rating - ratings.rating.mean()) / ratings.rating.std()

    # Sparse vector preparation
    user_sparse_vectors = defaultdict(lambda: {"values": [], "indices": []})
    for row in ratings.itertuples():
        user_sparse_vectors[row.userId]["values"].append(row.rating)
        user_sparse_vectors[row.userId]["indices"].append(row.movieId)

    # Upload data
    def data_generator():
        for user in ratings.itertuples():
            yield models.PointStruct(
                id=user.userId, vector={"ratings": user_sparse_vectors[user.userId]}, payload=user._asdict()
            )

    qdrant.upload_points("movielens", data_generator())
    return movies


# Function to input and normalize ratings
def input_ratings(movies):
    print("Enter ratings for the movies (scale 0 to 5, e.g., 'Black Panther, 5'):")
    print("Type 'done' when you are finished.")
    ratings = {}
    while True:
        entry = input("Enter movie and rating or type 'done' to finish: ")
        if entry.lower() == "done":
            break
        movie_name, rating = entry.rsplit(",", 1)
        movie_id = movies[movies.title.str.contains(movie_name.strip(), case=False)].movieId.iloc[0]
        normalized_rating = 2 * (float(rating) / 5) - 1  # Normalize from 0-5 to -1 to 1
        ratings[movie_id] = normalized_rating
    return ratings


# Search and recommendation function
def recommend_movies(qdrant, movies, my_ratings):
    def to_vector(ratings):
        vector = models.SparseVector(values=[], indices=[])
        for movieId, rating in ratings.items():
            vector.values.append(rating)
            vector.indices.append(movieId)
        return vector

    results = qdrant.search(
        "movielens",
        query_vector=models.NamedSparseVector(name="ratings", vector=to_vector(my_ratings)),
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
    print("Recommended movies for you:")

    for movieId, score in top_movies[:5]:
        print(movies[movies.movieId == movieId].title.values[0], score)


# Main execution
if __name__ == "__main__":
    qdrant = init_qdrant()
    movies = load_data(qdrant)
    my_ratings = input_ratings(movies)
    recommend_movies(qdrant, movies, my_ratings)
