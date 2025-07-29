from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list
import numpy as np
from scipy.stats import pearsonr
import random
import sys
import time
import os

def calculate_rating_correlation(user1_ratings, user2_ratings):
    """Calculate Pearson correlation between two users' ratings."""
    # Convert ratings to dictionaries for easy lookup
    user1_dict = {movie_id: rating for movie_id, rating in user1_ratings}
    user2_dict = {movie_id: rating for movie_id, rating in user2_ratings}
    
    # Find common movies
    common_movies = set(user1_dict.keys()).intersection(set(user2_dict.keys()))
    
    # If less than 5 movies in common, return None (not enough data)
    if len(common_movies) < 5:
        return None
    
    # Extract ratings for common movies
    u1_ratings = [user1_dict[m] for m in common_movies]
    u2_ratings = [user2_dict[m] for m in common_movies]
    
    # Calculate Pearson correlation
    try:
        corr, _ = pearsonr(u1_ratings, u2_ratings)
        return corr
    except:
        return None

if __name__ == "__main__":
    spark = SparkSession.builder.appName("MovieTwinsValidation").getOrCreate()
    sc = spark.sparkContext
    
    start = time.time()
    
    # Use absolute file paths with file:// protocol
    base_dir = "/home/jj3007_nyu_edu/capstone-bdcs-79"
    # Use HDFS paths
    ratings_path = "hdfs:///user/jj3007_nyu_edu/capstone-bdcs-79/ml-latest/ratings.csv"
    top_pairs_path = "hdfs:///user/jj3007_nyu_edu/capstone-bdcs-79/movie_twins_output_large.csv"
    
    print(f"Reading ratings from {ratings_path}")
    print(f"Reading top pairs from {top_pairs_path}")
    
    # Load ratings
    ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)
    
    # Load the previously calculated top 100 similar pairs
    top_pairs_df = spark.read.csv(top_pairs_path, header=True, inferSchema=True)
    
    # Convert to list of tuples [(user1, user2), ...]
    top_pairs = [(row["user1"], row["user2"]) for row in top_pairs_df.collect()]
    
    print(f"Loaded {len(top_pairs)} top pairs")
    
    # Get user ratings
    user_ratings = ratings_df.select("userId", "movieId", "rating") \
                            .rdd \
                            .map(lambda row: (row[0], (row[1], row[2]))) \
                            .groupByKey() \
                            .mapValues(list) \
                            .collectAsMap()
    
    print(f"Loaded ratings for {len(user_ratings)} users")
    
    # Calculate correlation for top pairs
    top_pair_correlations = []
    for user1, user2 in top_pairs:
        if user1 in user_ratings and user2 in user_ratings:
            corr = calculate_rating_correlation(user_ratings[user1], user_ratings[user2])
            if corr is not None:
                top_pair_correlations.append(corr)
    
    print(f"Calculated correlations for {len(top_pair_correlations)} valid top pairs")
    
    # Generate 100 random pairs
    all_users = list(user_ratings.keys())
    random_pairs = set()
    while len(random_pairs) < 100:
        u1 = random.choice(all_users)
        u2 = random.choice(all_users)
        if u1 != u2 and (u1, u2) not in random_pairs and (u2, u1) not in random_pairs:
            random_pairs.add((u1, u2))
    
    # Calculate correlation for random pairs
    random_pair_correlations = []
    for user1, user2 in random_pairs:
        corr = calculate_rating_correlation(user_ratings[user1], user_ratings[user2])
        if corr is not None:
            random_pair_correlations.append(corr)
    
    print(f"Calculated correlations for {len(random_pair_correlations)} valid random pairs")
    
    # Calculate and print average correlations
    avg_top_corr = np.mean(top_pair_correlations) if top_pair_correlations else 0
    avg_random_corr = np.mean(random_pair_correlations) if random_pair_correlations else 0
    
    print("\n=== VALIDATION RESULTS ===")
    print(f"Top similar pairs: Average correlation = {avg_top_corr:.4f} (from {len(top_pair_correlations)} valid pairs)")
    print(f"Random pairs: Average correlation = {avg_random_corr:.4f} (from {len(random_pair_correlations)} valid pairs)")
    print(f"Difference: {avg_top_corr - avg_random_corr:.4f}")
    
    # Perform a statistical test to check if the difference is significant
    from scipy.stats import ttest_ind
    if len(top_pair_correlations) > 0 and len(random_pair_correlations) > 0:
        t_stat, p_value = ttest_ind(top_pair_correlations, random_pair_correlations, equal_var=False)
        print(f"T-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        print(f"The difference is {'statistically significant' if p_value < 0.05 else 'not statistically significant'}")
    
    # Save the results to a local file
    with open(f"{base_dir}/correlation_validation_results.txt", "w") as f:
        f.write("=== VALIDATION RESULTS ===\n")
        f.write(f"Top similar pairs: Average correlation = {avg_top_corr:.4f} (from {len(top_pair_correlations)} valid pairs)\n")
        f.write(f"Random pairs: Average correlation = {avg_random_corr:.4f} (from {len(random_pair_correlations)} valid pairs)\n")
        f.write(f"Difference: {avg_top_corr - avg_random_corr:.4f}\n")
        if len(top_pair_correlations) > 0 and len(random_pair_correlations) > 0:
            f.write(f"T-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}\n")
            f.write(f"The difference is {'statistically significant' if p_value < 0.05 else 'not statistically significant'}\n")
    
    print(f"\nResults saved to {base_dir}/correlation_validation_results.txt")
    print(f"\nDone in {time.time() - start:.2f} seconds")
    spark.stop()