from pyspark.sql import SparkSession
from datasketch import MinHash, MinHashLSH
import time

def load_user_movie_sets_spark(spark, ratings_path):
    df = spark.read.csv(ratings_path, header=True, inferSchema=True)
    user_movie_rdd = df.rdd.map(lambda row: (row["userId"], row["movieId"]))
    user_movie_sets = user_movie_rdd.groupByKey().mapValues(set).collectAsMap()
    return user_movie_sets

def create_minhash(movie_set, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for movie_id in movie_set:
        m.update(str(movie_id).encode('utf8'))
    return m

def compute_similar_pairs(user_movie_sets, num_perm=128, top_n=100):
    print("Creating MinHash signatures...")
    user_minhashes = {
        user: create_minhash(movie_set, num_perm)
        for user, movie_set in user_movie_sets.items()
    }

    lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
    for user, m in user_minhashes.items():
        lsh.insert(str(user), m)

    print("Finding similar pairs...")
    pairs = set()
    similarities = []

    for user, m in user_minhashes.items():
        similar_users = lsh.query(m)
        for other in similar_users:
            if str(user) == str(other):
                continue  
            uid, oid = int(user), int(other)
            if uid > oid:
                continue
            jaccard = m.jaccard(user_minhashes[oid])
            pairs.add((uid, oid))
            similarities.append((uid, oid, jaccard))

    top_pairs = sorted(similarities, key=lambda x: x[2], reverse=True)[:top_n]
    return top_pairs

if __name__ == "__main__":
    spark = SparkSession.builder.appName("MovieTwins").getOrCreate()

    start = time.time()
    path = "hdfs:///user/rg4930_nyu_edu/ml-latest/ratings.csv"

    user_movie_sets = load_user_movie_sets_spark(spark, path)
    print(f"Loaded {len(user_movie_sets)} users")

    top_pairs = compute_similar_pairs(user_movie_sets)
    print("\nTop 10 similar pairs:")
    for uid, oid, sim in top_pairs[:10]:
        print(f"User {uid} & User {oid} -> Similarity: {sim:.3f}")
    result_df = spark.createDataFrame(top_pairs, ["user1", "user2", "similarity"])
    output_path = "hdfs:///user/rg4930_nyu_edu/movie_twins_output_large"
    result_df.write.mode("overwrite").csv(output_path, header=True)

    print(f"\nDone in {time.time() - start:.2f} seconds")
    spark.stop()

