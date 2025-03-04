from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("MergeHDFSFiles") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

# Function to merge part files in HDFS into a single CSV
def merge_hdfs_csv(hdfs_dir, output_file):
    df = spark.read.csv(f"hdfs://localhost:9000/{hdfs_dir}", header=True, inferSchema=True)
    df.toPandas().to_csv(f"{output_file}", index=False)
    print(f"Merged {hdfs_dir} into {output_file}")

# Merge all HDFS directories into single CSV files
merge_hdfs_csv("movies", "movies.csv")
merge_hdfs_csv("ratings", "ratings.csv")
merge_hdfs_csv("predicted_ratings", "predicted_ratings.csv")
