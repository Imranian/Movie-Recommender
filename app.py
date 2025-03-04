from flask import Flask, jsonify, request, render_template
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np

app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("MovieRecommendation").getOrCreate()

# Function to read merged CSV files from HDFS
def read_hdfs_csv(hdfs_path):
    return spark.read.option("header", "true").csv(f"hdfs://localhost:9000{hdfs_path}")

# Load data from HDFS into Pandas DataFrames
movies_df = read_hdfs_csv("/merged_movies.csv").toPandas()
ratings_df = read_hdfs_csv("/merged_ratings.csv").toPandas()
predicted_ratings_df = read_hdfs_csv("/merged_predicted_ratings.csv").toPandas()

# Convert predicted ratings to numeric
predicted_ratings_df.iloc[:, 1:] = predicted_ratings_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0)

# Ensure `predicted_ratings_df` has the same shape as `movies_df`
num_movies = len(movies_df)
num_predicted_movies = predicted_ratings_df.shape[1] - 1

# If mismatch, adjust by padding missing columns with zeros
if num_predicted_movies < num_movies:
    missing_cols = num_movies - num_predicted_movies
    padding = np.zeros((len(predicted_ratings_df), missing_cols))
    predicted_ratings_df = pd.concat(
        [predicted_ratings_df, pd.DataFrame(padding, columns=[f"missing_{i}" for i in range(missing_cols)])],
        axis=1
    )
elif num_predicted_movies > num_movies:
    predicted_ratings_df = predicted_ratings_df.iloc[:, :num_movies+1]  # Trim extra columns

# Debugging: Check shape
print(f"Movies Count: {num_movies}, Predicted Ratings Shape: {predicted_ratings_df.shape}")

# Store user clicks
user_clicks = {}

@app.route('/')
def home():
    """ Serve the homepage """
    return render_template("index.html")

@app.route('/movies', methods=['GET'])
def get_movies():
    """ Fetch up to 50 movies from HDFS without errors. """
    fresh_movies_df = read_hdfs_csv("/merged_movies.csv").toPandas()  # Reload from HDFS
    
    # Ensure we don't request more samples than available
    num_movies = min(50, len(fresh_movies_df))  # Take the smaller value
    sampled_movies = fresh_movies_df.sample(n=num_movies, random_state=42)  # Adjust sample size dynamically
    
    return jsonify(sampled_movies.to_dict(orient='records'))

@app.route('/click', methods=['POST'])
def log_click():
    """ Store movie clicks for the user """
    user_id = str(request.json.get("user_id"))
    movie_id = request.json.get("movie_id")

    if user_id not in user_clicks:
        user_clicks[user_id] = []

    user_clicks[user_id].append(movie_id)
    print(f"User {user_id} clicked: {user_clicks[user_id]}")  # Debugging

    return jsonify({"message": "Click recorded", "user_clicks": user_clicks})

@app.route('/recommend', methods=['GET'])
def recommend_movies():
    """ Generate recommendations based on user clicks """
    user_id = str(request.args.get("user_id"))

    if user_id not in user_clicks or len(user_clicks[user_id]) == 0:
        return jsonify({"message": "No interactions found. Showing top-rated movies.", "recommended_movies": movies_df["title"].head(5).tolist()})

    clicked_movies = user_clicks[user_id]
    clicked_movie_indices = movies_df[movies_df["id"].astype(str).isin(clicked_movies)].index.tolist()

    print(f"User {user_id} clicked: {clicked_movies} (indices: {clicked_movie_indices})")

    if not clicked_movie_indices:
        return jsonify({"recommended_movies": movies_df["title"].head(5).tolist()})

    # Get predicted ratings for all movies
    avg_predicted_scores = np.zeros(len(movies_df))

    for movie_index in clicked_movie_indices:
        movie_ratings = predicted_ratings_df.iloc[:, 1:].iloc[movie_index].values.astype(float)

        # Ensure correct shape
        if movie_ratings.shape[0] != avg_predicted_scores.shape[0]:
            movie_ratings = np.pad(movie_ratings, (0, avg_predicted_scores.shape[0] - movie_ratings.shape[0]), mode='constant')

        avg_predicted_scores += movie_ratings

    avg_predicted_scores /= len(clicked_movie_indices)

    # Recommend top 5 movies based on recalculated scores
    top_indices = np.argsort(avg_predicted_scores)[-5:][::-1]
    recommended_movies = movies_df.iloc[top_indices]["title"].values.tolist()

    print(f"Recommended Movies for User {user_id}: {recommended_movies}")

    return jsonify({"recommended_movies": recommended_movies})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
