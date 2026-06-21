import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings


def get_top_movies(movies, ratings, n=10):
    movie_avg_rating = (
        ratings.groupby("movieId")["rating"]
        .mean()
        .reset_index()
    )

    movie_avg_rating = movie_avg_rating.merge(
        movies,
        on="movieId"
    )

    top_movies = movie_avg_rating.sort_values(
        by="rating",
        ascending=False
    )

    return top_movies[["title", "rating"]].head(n)


def build_item_similarity(ratings):
    user_item_matrix = ratings.pivot(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    item_similarity = cosine_similarity(
        user_item_matrix.T
    )

    return user_item_matrix, item_similarity


def similar_movies(movie_id, movies, user_item_matrix, item_similarity, top_n=5):
    movie_ids = list(user_item_matrix.columns)

    if movie_id not in movie_ids:
        return "Movie not found."

    idx = movie_ids.index(movie_id)

    similarity_scores = list(
        enumerate(item_similarity[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similar_ids = []

    for i, score in similarity_scores[1:top_n + 1]:
        similar_ids.append(movie_ids[i])

    return movies[
        movies["movieId"].isin(similar_ids)
    ][["movieId", "title"]]


# -----------------------------
# Main Program
# -----------------------------

movies, ratings = load_data()

user_item_matrix, item_similarity = build_item_similarity(
    ratings
)

print("\n🔥 Top Movies")
print(get_top_movies(movies, ratings))

print("\n🎬 Movies Similar To movieId=1")
print(
    similar_movies(
        movie_id=1,
        movies=movies,
        user_item_matrix=user_item_matrix,
        item_similarity=item_similarity
    )
)