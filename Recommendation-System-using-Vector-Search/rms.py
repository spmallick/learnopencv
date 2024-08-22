import pandas as pd
from qdrant_client import QdrantClient, models
from collections import defaultdict
import numpy as np

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
    # Normalize ratings
    ratings['normalized_rating'] = (ratings.rating - ratings.rating.mean(axis=0)) / ratings.rating.std()

    # Sparse vector preparation
    user_sparse_vectors = defaultdict(lambda: {"values": [], "indices": []})
    for row in ratings.itertuples():
        user_sparse_vectors[row.userId]["values"].append(row.normalized_rating)
        user_sparse_vectors[row.userId]["indices"].append(row.movieId)
        
    # Calculate total number of key-value pairs and number of users
    total_key_value_pairs = sum(len(v['values']) for v in user_sparse_vectors.values())
    total_users = len(user_sparse_vectors)
    
    print(f"Total number of users: {total_users}")
    print(f"Total number of key-value pairs: {total_key_value_pairs}")

    # Upload data
    def data_generator():
        for user_id, vector in user_sparse_vectors.items():
            yield models.PointStruct(
                id=user_id, vector={"ratings": vector}, payload={}
            )

    qdrant.upload_points("movielens", data_generator())

# Function to retrieve and print vectors from Qdrant
def print_uploaded_vectors(qdrant):
    # Retrieve all points from the collection
    results = qdrant.scroll(collection_name="movielens", limit=10, with_vectors=True)

    print("Uploaded sparse vectors and their shapes from Qdrant:")
    for result in results[0]:
        user_id = result.id
        user_vector = result.vector["ratings"]
        
        # Accessing the values and indices directly from the SparseVector object
        vector_shape = (len(user_vector.values), len(user_vector.indices))
        print(f"User ID: {user_id}, Sparse Vector Shape: {vector_shape}, Sparse Vector: {{'values': {user_vector.values}, 'indices': {user_vector.indices}}}")
        
# Function to input and normalize ratings
def input_ratings(movies, ratings):
    print("Enter ratings for the movies (scale 0 to 5, e.g., 'Black Panther, 5'):")
    print("Type 'done' when you are finished.")
    final_ratings = {}
    
    mean_rating = ratings.rating.mean()
    std_rating = ratings.rating.std()
    
    while True:
        entry = input("Enter movie and rating or type 'done' to finish: ")
        if entry.lower() == "done":
            break
        try:
            movie_name, user_rating = entry.rsplit(",", 1)
            user_rating = float(user_rating.strip())
            movie_id = movies[movies.title.str.contains(movie_name.strip(), case=False)].movieId.iloc[0]
            
            # Normalize the user's rating
            normalized_input_rating = (user_rating - mean_rating) / std_rating
            print('normalized rating:', normalized_input_rating)
            final_ratings[movie_id] = normalized_input_rating
        except IndexError:
            print(f"Movie '{movie_name.strip()}' not found in the database. Please try again.")
        except ValueError:
            print("Please enter a valid rating between 0 and 5.")
    
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
    results = np.array(results)
    print(results.shape)
    #print(results)
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
    load_data(qdrant)  # This only uploads data, no return needed
    print_uploaded_vectors(qdrant)
    my_ratings = input_ratings(movies, ratings)
    recommend_movies(qdrant, movies, my_ratings)