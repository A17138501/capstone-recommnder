from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list
import numpy as np
from scipy.stats import pearsonr, ttest_ind
import random
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
    
    # Check if either array is constant (all values the same)
    if len(set(u1_ratings)) == 1 or len(set(u2_ratings)) == 1:
        return None  # Can't calculate correlation for constant arrays
    
    # Calculate Pearson correlation
    try:
        corr, _ = pearsonr(u1_ratings, u2_ratings)
        if np.isnan(corr):
            return None
        return corr
    except:
        return None

def process_batch(batch_pairs, ratings_df, spark):
    """Process a batch of user pairs and calculate correlations."""
    # Extract unique users in this batch
    batch_users = set()
    for user1, user2 in batch_pairs:
        batch_users.add(user1)
        batch_users.add(user2)
    
    batch_users_list = list(batch_users)
    
    # Filter ratings to only include users in this batch
    batch_ratings_df = ratings_df.filter(col("userId").isin(batch_users_list))
    
    # Get user ratings for this batch only
    user_ratings = batch_ratings_df.select("userId", "movieId", "rating") \
                        .rdd \
                        .map(lambda row: (row[0], (row[1], row[2]))) \
                        .groupByKey() \
                        .mapValues(list) \
                        .collectAsMap()
    
    # Calculate correlation for pairs in this batch
    batch_correlations = []
    for user1, user2 in batch_pairs:
        if user1 in user_ratings and user2 in user_ratings:
            corr = calculate_rating_correlation(user_ratings[user1], user_ratings[user2])
            if corr is not None:
                batch_correlations.append(corr)
    
    return batch_correlations

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
    
    # Process top pairs in batches
    batch_size = 20  # Adjust based on your memory constraints
    top_pair_correlations = []
    
    for i in range(0, len(top_pairs), batch_size):
        batch = top_pairs[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(top_pairs) + batch_size - 1)//batch_size} (top pairs)")
        batch_results = process_batch(batch, ratings_df, spark)
        top_pair_correlations.extend(batch_results)
    
    # Filter out any None or NaN values
    top_pair_correlations = [c for c in top_pair_correlations if c is not None and not np.isnan(c)]
    print(f"Calculated correlations for {len(top_pair_correlations)} valid top pairs")
    
    # Get all users for random selection
    all_users = ratings_df.select("userId").distinct().rdd.map(lambda r: r[0]).collect()

    # Instead of generating a fixed number of random pairs upfront,
    # keep generating and processing until we have 100 valid correlations
    random_pair_correlations = []
    random_pairs_processed = 0
    batch_size = 20  # Maintain the batch processing for efficiency

    print("\nGenerating random pairs until we have 100 valid correlations...")
    while len(random_pair_correlations) < 100:
        # Generate a new batch of random pairs
        new_random_pairs = set()
        while len(new_random_pairs) < batch_size:
            u1 = random.choice(all_users)
            u2 = random.choice(all_users)
            if u1 != u2 and (u1, u2) not in new_random_pairs and (u2, u1) not in new_random_pairs:
                new_random_pairs.add((u1, u2))
        
        # Process this batch
        batch_results = process_batch(list(new_random_pairs), ratings_df, spark)
        
        # Filter out None and NaN values and add to our collection
        valid_results = [c for c in batch_results if c is not None and not np.isnan(c)]
        random_pair_correlations.extend(valid_results)
        
        # Update progress
        random_pairs_processed += batch_size
        print(f"Processed {random_pairs_processed} random pairs, found {len(random_pair_correlations)} valid correlations")
        
        # Optional: avoid infinite loops if valid pairs are too rare
        if random_pairs_processed > 10000:
            print("WARNING: Processed over 10,000 random pairs without finding 100 valid correlations.")
            print(f"Proceeding with {len(random_pair_correlations)} valid random pairs.")
            break

    # Trim to exactly 100 if we got more
    if len(random_pair_correlations) > 100:
        random_pair_correlations = random_pair_correlations[:100]

    print(f"Collected {len(random_pair_correlations)} valid random pairs after processing {random_pairs_processed} total random pairs")
    
    # Add diagnostics to understand the data better
    print("\n=== DIAGNOSTIC INFO ===")
    print(f"Top pairs max correlation: {max(top_pair_correlations) if top_pair_correlations else 'N/A'}")
    print(f"Top pairs min correlation: {min(top_pair_correlations) if top_pair_correlations else 'N/A'}")
    print(f"Random pairs max correlation: {max(random_pair_correlations) if random_pair_correlations else 'N/A'}")
    print(f"Random pairs min correlation: {min(random_pair_correlations) if random_pair_correlations else 'N/A'}")
    
    # Calculate and print average correlations
    avg_top_corr = np.mean(top_pair_correlations) if top_pair_correlations else 0
    avg_random_corr = np.mean(random_pair_correlations) if random_pair_correlations else 0
    
    print("\n=== VALIDATION RESULTS ===")
    print(f"Top similar pairs: Average correlation = {avg_top_corr:.4f} (from {len(top_pair_correlations)} valid pairs)")
    print(f"Random pairs: Average correlation = {avg_random_corr:.4f} (from {len(random_pair_correlations)} valid pairs)")
    print(f"Difference: {avg_top_corr - avg_random_corr:.4f}")
    
    # Perform a statistical test to check if the difference is significant
    if len(top_pair_correlations) > 0 and len(random_pair_correlations) > 0:
        try:
            t_stat, p_value = ttest_ind(top_pair_correlations, random_pair_correlations, equal_var=False)
            print(f"T-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
            print(f"The difference is {'statistically significant' if p_value < 0.05 else 'not statistically significant'}")
        except Exception as e:
            print(f"Error in t-test: {str(e)}")
    else:
        print("Cannot perform t-test: insufficient valid correlation data")
    
    # Save the results to a local file
    with open(f"{base_dir}/correlation_validation_results.txt", "w") as f:
        f.write("=== VALIDATION RESULTS ===\n")
        f.write(f"Top similar pairs: Average correlation = {avg_top_corr:.4f} (from {len(top_pair_correlations)} valid pairs)\n")
        f.write(f"Random pairs: Average correlation = {avg_random_corr:.4f} (from {len(random_pair_correlations)} valid pairs)\n")
        f.write(f"Difference: {avg_top_corr - avg_random_corr:.4f}\n")
        if len(top_pair_correlations) > 0 and len(random_pair_correlations) > 0:
            try:
                f.write(f"T-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}\n")
                f.write(f"The difference is {'statistically significant' if p_value < 0.05 else 'not statistically significant'}\n")
            except:
                f.write("Could not perform t-test\n")
    
    print(f"\nResults saved to {base_dir}/correlation_validation_results.txt")
    print(f"\nDone in {time.time() - start:.2f} seconds")
    spark.stop()