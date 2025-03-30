# Movie Recommender

## Using Autoencoders on Real-Time Data to Recommend Movies Based on User Interaction

---

## Introduction
Movie Recommender is a deep learning-powered web application that provides personalized movie recommendations based on user interactions.

## Objectives
1. Use TMDB Database for data ingestion.
2. Implement Big Data Technologies like Spark and Hadoop for data processing.
3. Implement a deep learning model (autoencoder) on the movie data.
4. Develop a web page named *Movie Recommender* using Flask.

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Flask | Backend framework |
| Spark and HDFS (Hadoop) | Distributed computing and storage |
| TensorFlow & Autoencoders | Deep learning model |
| TMDB API | Fetching real-time movie data |
| Pandas & NumPy | Data manipulation and processing |
| HTML, CSS | Frontend UI |

## Data Pipeline

![Figure1: Data Pipeline](https://github.com/Imranian/Movie-Recommender/blob/401dd084261816fd5476be105d7b6095d17c3527/Data_Pipeline.drawio.png)

### 1. Data Ingestion/Streaming
- Data related to recent movies is extracted from the TMDB database using an API key.

### 2. Data Processing
- Data is pre-processed and manipulated using Apache Spark and stored in HDFS (Hadoop Distributed File System).

### 3. Deep Learning Model
- The data from HDFS is used to train the deep learning model (autoencoder).

### 4. Web Application
- A web page is created using HTML, CSS, and Flask, displaying movies for user interaction.


![Recommended Movies](https://github.com/Imranian/Movie-Recommender/blob/401dd084261816fd5476be105d7b6095d17c3527/DL%20project.png)
