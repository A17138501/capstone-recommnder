from pyspark.sql import SparkSession
from pyspark.sql.functions import count, avg, col, collect_list, lit
from pyspark.ml.evaluation import RankingEvaluator
import argparse

def main():
    """
    Implement and evaluate a popularity baseline model for movie recommendations.
    The model recommends the same top N most popular movies to all users.
    """
    parser = argparse.ArgumentParser(description="Popularity baseline model")
    parser.add_argument("--train_path", required=True, help="HDFS path to training data")
    parser.add_argument("--val_path", required=True, help="HDFS path to validation data")
    parser.add_argument("--num_recommendations", type=int, default=100, 
                        help="Number of movies to recommend (default: 100)")
    args = parser.parse_args()

    # Initialize Spark
    spark = SparkSession.builder.appName("PopularityBaseline").getOrCreate()
    
    # Load training and validation data
    print("Loading training and validation data...")
    train = spark.read.csv(args.train_path, header=True, inferSchema=True)
    val = spark.read.csv(args.val_path, header=True, inferSchema=True)
    
    # Calculate movie popularity based on number of ratings in training set
    print("Building popularity baseline model...")
    movie_popularity = train.groupBy("movieId").agg(
        count("rating").alias("rating_count"),
        avg("rating").alias("avg_rating")
    )
    
    # Sort by count (most popular first)
    popular_movies = movie_popularity.orderBy(col("rating_count").desc())
    
    # Take top N movies as recommendations
    top_movies = popular_movies.limit(args.num_recommendations).select("movieId")
    top_movie_ids = [float(row.movieId) for row in top_movies.collect()]  # Convert to float/double
    
    print(f"Top {args.num_recommendations} most popular movies: {top_movie_ids[:5]}...")
    
    # Get distinct users from validation set
    val_users = val.select("userId").distinct()
    
    # Create a fixed array of top movie IDs for recommendations
    rec_array = top_movie_ids
    
    # Create recommendations for all users (same list for everyone)
    user_recommendations = val_users.withColumn(
        "prediction", lit(rec_array)
    )
    
    # Create ground truth from validation set
    # Convert movieId to double in the collect_list
    ground_truth = val.selectExpr("userId", "CAST(movieId AS DOUBLE) as movieId").groupBy("userId").agg(
        collect_list("movieId").alias("label")
    )
    
    # Join predictions with ground truth
    predictions_and_labels = user_recommendations.join(
        ground_truth, on="userId", how="inner"
    ).select("prediction", "label")
    
    # Print schema to verify data types
    print("Evaluation dataset schema:")
    predictions_and_labels.printSchema()
    
    # Show sample data for verification
    print("Sample evaluation data:")
    predictions_and_labels.show(5, truncate=False)
    
    # Count total evaluation pairs
    eval_count = predictions_and_labels.count()
    print(f"Total user-movie pairs for evaluation: {eval_count}")
    
    # Define evaluators
    evaluators = {
        "precisionAt10": RankingEvaluator(k=10, metricName="precisionAtK"),
        "ndcgAt10": RankingEvaluator(k=10, metricName="ndcgAtK"),
        "mapAt10": RankingEvaluator(k=10, metricName="meanAveragePrecisionAtK"),
        "precisionAt100": RankingEvaluator(k=100, metricName="precisionAtK"),
        "ndcgAt100": RankingEvaluator(k=100, metricName="ndcgAtK"),
        "mapAt100": RankingEvaluator(k=100, metricName="meanAveragePrecisionAtK")
    }
    
    # Calculate each metric
    print("Evaluating baseline model...")
    metrics = {}
    for name, evaluator in evaluators.items():
        try:
            metrics[name] = evaluator.evaluate(predictions_and_labels)
            print(f"Successfully calculated {name}")
        except Exception as e:
            print(f"Error calculating {name}: {e}")
            metrics[name] = 0.0
    
    # Print results
    print("\nBaseline Model Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save the model (top movie IDs) for future use
    with open("popularity_baseline_model.txt", "w") as f:
        f.write(",".join(map(str, top_movie_ids)))
    print("Model saved as popularity_baseline_model.txt")
    
    spark.stop()

if __name__ == "__main__":
    main()