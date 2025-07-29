from pyspark.sql import SparkSession
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition ratings data into train/val/test CSVs")
    parser.add_argument("--ratings_path", required=True, help="HDFS path to ratings.csv")
    parser.add_argument("--train_path", required=True, help="HDFS output path for training set")
    parser.add_argument("--val_path", required=True, help="HDFS output path for validation set")
    parser.add_argument("--test_path", required=True, help="HDFS output path for test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Initialize Spark
    spark = SparkSession.builder.appName("PartitionRatings").getOrCreate()

    # Load full ratings data
    ratings = spark.read.csv(args.ratings_path, header=True, inferSchema=True)

    # Random split into 60% train, 20% val, 20% test
    train, val, test = ratings.randomSplit([0.6, 0.2, 0.2], seed=args.seed)

    # Save to HDFS as CSV (could also use .parquet if needed)
    train.write.mode("overwrite").csv(args.train_path, header=True)
    val.write.mode("overwrite").csv(args.val_path, header=True)
    test.write.mode("overwrite").csv(args.test_path, header=True)

    print("Partitioning complete and saved to HDFS.")
