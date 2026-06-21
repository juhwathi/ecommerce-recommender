import pandas as pd

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

def get_top_movies(n=10):
    movie_avg_rating = ratings.groupby("movieId")["rating"].mean().reset_index()
    movie_avg_rating = movie_avg_rating.merge(movies, on="movieId")

    top_movies = movie_avg_rating.sort_values("rating", ascending=False)

    return top_movies[["title", "rating"]].head(n)
user_item_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

item_similarity = cosine_similarity(
    user_item_matrix.T
)
print("\nUser-Item Matrix shape:")
print(user_item_matrix.shape)

print("\n🔥 Top Movies:")
print(get_top_movies(10))