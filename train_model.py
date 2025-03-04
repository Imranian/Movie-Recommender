import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder \
    .appName("MovieRecommendation") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

API_KEY = "43d64229413cc934538918dd45a80f87"
BASE_URL = "https://api.themoviedb.org/3"

# Function to fetch movies from TMDB
def fetch_movies():
    url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page=1"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["results"]
    return []

# Fetch and preprocess movie data
movies = fetch_movies()
df_movies = pd.DataFrame(movies)[["id", "title", "vote_average", "vote_count"]]

# Convert to Spark DataFrame and save to HDFS
spark_df_movies = spark.createDataFrame(df_movies)
spark_df_movies.write.mode("overwrite").csv("hdfs://localhost:9000/movies", header=True)

# Simulate user ratings
num_users = 100
num_movies = len(df_movies)
ratings_matrix = np.random.randint(0, 6, size=(num_users, num_movies))  # Ratings from 0 to 5
df_ratings = pd.DataFrame(ratings_matrix, columns=df_movies["title"])

# Convert to Spark DataFrame and save to HDFS
spark_df_ratings = spark.createDataFrame(df_ratings)
spark_df_ratings.write.mode("overwrite").csv("hdfs://localhost:9000/ratings", header=True)

# Load data from HDFS and preprocess
movies_df = spark.read.csv("hdfs://localhost:9000/movies", header=True, inferSchema=True)
ratings_df = spark.read.csv("hdfs://localhost:9000/ratings", header=True, inferSchema=True)

# Convert to Pandas for deep learning
ratings_pd = ratings_df.toPandas()
scaler = MinMaxScaler()
ratings_scaled = scaler.fit_transform(ratings_pd)

# Train Autoencoder
train_data, test_data = train_test_split(ratings_scaled, test_size=0.2, random_state=42)
input_dim = num_movies

autoencoder = keras.models.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(input_dim, activation='sigmoid')  # Output same shape as input
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(train_data, train_data, epochs=50, batch_size=16, validation_data=(test_data, test_data))

# Predict user ratings
predicted_ratings = autoencoder.predict(ratings_scaled)
predicted_ratings = scaler.inverse_transform(predicted_ratings)

# Convert to Spark DataFrame
predicted_df = pd.DataFrame(predicted_ratings, columns=df_movies["title"])
spark_df_predicted = spark.createDataFrame(predicted_df)

# Save predictions to HDFS
spark_df_predicted.write.mode("overwrite").csv("hdfs://localhost:9000/predicted_ratings", header=True)

print("All data saved to HDFS!")

# Function to merge HDFS CSV parts into a single file
def merge_hdfs_csv(hdfs_dir, output_file):
    df = spark.read.csv(f"hdfs://localhost:9000/{hdfs_dir}", header=True, inferSchema=True)
    df.toPandas().to_csv(f"{output_file}", index=False)
    print(f"Merged {hdfs_dir} into {output_file}")

# Merge all HDFS directories into single CSV files
merge_hdfs_csv("movies", "movies.csv")
merge_hdfs_csv("ratings", "ratings.csv")
merge_hdfs_csv("predicted_ratings", "predicted_ratings.csv")

