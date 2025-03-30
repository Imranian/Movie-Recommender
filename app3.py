from flask import Flask, jsonify, request, render_template
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import requests
import os
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("MovieRecommendation").getOrCreate()

# TMDB API Configuration
API_KEY = "API KEY"
BASE_URL = "https://api.themoviedb.org/3"

def fetch_latest_movies():
    """ Fetch the latest movies from TMDB API dynamically """
    url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page=1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["results"]
    return []

@app.route('/movies', methods=['GET'])
def get_movies():
    """ Fetch fresh movies directly from TMDB API """
    fresh_movies = fetch_latest_movies()
    df_movies = pd.DataFrame(fresh_movies)[["id", "title", "vote_average", "vote_count"]]
    
    return jsonify(df_movies.to_dict(orient='records'))

# Function to read merged CSV files from HDFS
def read_hdfs_csv(hdfs_path):
    return spark.read.option("header", "true").csv(f"hdfs://localhost:9000{hdfs_path}")

# Load merged data from HDFS into Pandas DataFrames
movies_df = pd.DataFrame(fetch_latest_movies())  # Fetch latest movies from TMDB
movies_df = movies_df[["id", "title", "vote_average", "vote_count"]]  # Keep needed columns
movies_df["id"] = movies_df["id"].astype(str)  # Ensure IDs are strings
ratings_df = read_hdfs_csv("/merged_ratings.csv").toPandas()
predicted_ratings_df = read_hdfs_csv("/merged_predicted_ratings.csv").toPandas()

# Convert predicted ratings to numeric
predicted_ratings_df.iloc[:, 1:] = predicted_ratings_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0)

# Ensure predicted ratings match movies count
num_movies = len(movies_df)
num_predicted_movies = predicted_ratings_df.shape[1] - 1

if num_predicted_movies < num_movies:
    missing_cols = num_movies - num_predicted_movies
    padding = np.zeros((len(predicted_ratings_df), missing_cols))
    predicted_ratings_df = pd.concat(
        [predicted_ratings_df, pd.DataFrame(padding, columns=[f"missing_{i}" for i in range(missing_cols)])],
        axis=1
    )
elif num_predicted_movies > num_movies:
    predicted_ratings_df = predicted_ratings_df.iloc[:, :num_movies+1]  # Trim extra columns

print(f"Movies Count: {num_movies}, Predicted Ratings Shape: {predicted_ratings_df.shape}")

# Store user clicks
user_clicks = {}

@app.route('/')
def home():
    """ Serve the homepage """
    return render_template("index.html")

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
    clicked_movie_indices = movies_df[movies_df["id"].astype(str).isin(map(str, clicked_movies))].index.tolist()    
    print(f"Clicked Movies Found in DataFrame: {clicked_movie_indices}")
    print(f"Movies DF IDs: {movies_df['id'].tolist()[:10]}")  # Print first 10 movie IDs

    print(f"User {user_id} clicked: {clicked_movies} (indices: {clicked_movie_indices})")

    if not clicked_movie_indices:
        return jsonify({"recommended_movies": movies_df["title"].head(5).tolist()})

    # Get predicted ratings for all movies
    avg_predicted_scores = np.zeros(len(movies_df))

    for movie_index in clicked_movie_indices:
        movie_ratings = predicted_ratings_df.iloc[:, 1:].iloc[movie_index].values.astype(float)

        if movie_ratings.shape[0] != avg_predicted_scores.shape[0]:
            movie_ratings = np.pad(movie_ratings, (0, avg_predicted_scores.shape[0] - movie_ratings.shape[0]), mode='constant')

        avg_predicted_scores += movie_ratings

    avg_predicted_scores /= len(clicked_movie_indices)

    # Recommend top 5 movies based on recalculated scores
    top_indices = np.argsort(avg_predicted_scores)[-5:][::-1]
    recommended_movies = movies_df.iloc[top_indices]["title"].values.tolist()

    print(f"Recommended Movies for User {user_id}: {recommended_movies}")

    return jsonify({"recommended_movies": recommended_movies})

# Automatic Model Training Every 24 Hours
def train_model():
    """ Run the training script automatically every 24 hours """
    os.system("python train_model.py")

# Start background scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(train_model, "interval", hours=24)
scheduler.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
