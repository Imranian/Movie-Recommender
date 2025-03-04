import subprocess
import os
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("MergeHDFSFiles").getOrCreate()

def run_shell_command(command):
    """Runs a shell command and prints the output."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if stdout:
        print(stdout.decode())
    if stderr:
        print(stderr.decode())

def merge_csv_files(hdfs_path, output_file):
    """
    Merges multiple CSV files from an HDFS directory into a single CSV file.
    """
    full_hdfs_path = f"hdfs://localhost:9000/{hdfs_path}/*"
    
    # Check if HDFS path exists and has files
    print(f"Checking HDFS path: {full_hdfs_path}")
    list_output = subprocess.getoutput(f"hdfs dfs -ls {full_hdfs_path}")
    if "No such file or directory" in list_output or "0 B" in list_output:
        print(f"❌ ERROR: No data found in {hdfs_path}. Skipping merge.")
        return
    
    # Read CSV from HDFS
    df = spark.read.option("header", "true").csv(full_hdfs_path)
    
    # Save as a single CSV file (temporarily in local filesystem)
    temp_output_dir = "/tmp/temp_output"
    
    # Ensure directory is empty before writing
    if os.path.exists(temp_output_dir):
        run_shell_command(f"rm -r {temp_output_dir}")
    
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(temp_output_dir)
    
    # Find the actual CSV file inside the temporary directory
    try:
        part_files = [f for f in os.listdir(temp_output_dir) if f.startswith("part-")]
        if not part_files:
            print(f"❌ ERROR: No part files found in {temp_output_dir}.")
            return
        part_file = part_files[0]
    except Exception as e:
        print(f"❌ ERROR: Failed to list files in {temp_output_dir}. {str(e)}")
        return

    output_local = f"/tmp/{output_file}.csv"
    os.rename(f"{temp_output_dir}/{part_file}", output_local)  # Rename to a proper file name

    # Create directory in HDFS and move merged file
    hdfs_output_path = f"hdfs://localhost:9000/{output_file}"
    run_shell_command(f"hdfs dfs -rm -r {hdfs_output_path}")  # Remove previous files
    run_shell_command(f"hdfs dfs -mkdir -p {hdfs_output_path}")
    run_shell_command(f"hdfs dfs -put {output_local} {hdfs_output_path}/merged.csv")
    
    print(f"✅ Merged file saved to: {hdfs_output_path}/merged.csv")

# Merge datasets
merge_csv_files("movies", "merged_movies")
merge_csv_files("ratings", "merged_ratings")
merge_csv_files("predicted_ratings", "merged_predicted_ratings")
