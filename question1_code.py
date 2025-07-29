from pyspark.sql import SparkSession
from datasketch import MinHash, MinHashLSH
import sys
import time

NUM_PERM = 256
TOP_N = 100
LSH_THRESHOLD = 0.8
BATCH_SIZE = 10000

def create_minhash(movie_ids):
    m = MinHash(num_perm=NUM_PERM)
    for mid in movie_ids:
        m.update(str(mid).encode('utf8'))
    return m

def compute_batch(batch_items):
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    sig_dict = {}
    sims = []

    for uid, m in batch_items:
        lsh.insert(str(uid), m)
        sig_dict[uid] = m

    for uid, m in batch_items:
        for other in lsh.query(m):
            oid = int(other)
            if uid >= oid:
                continue
            jaccard = m.jaccard(sig_dict[oid])
            sims.append((uid, oid, jaccard))

    return sims

if __name__ == "__main__":
    spark = SparkSession.builder.appName("MovieTwins-Optimized").getOrCreate()
    sc = spark.sparkContext
    start = time.time()

    path = "hdfs:///user/rg4930_nyu_edu/ml-latest/ratings.csv"
    df = spark.read.csv(path, header=True, inferSchema=True)

    user_movies = df.rdd.map(lambda row: (row[0], row[1])).groupByKey().mapValues(set).filter(lambda x: len(x[1]) >= 20)

    user_sigs = user_movies.mapValues(create_minhash)

    indexed = user_sigs.zipWithIndex().map(lambda x: (x[1] // BATCH_SIZE, x[0]))
    batches = indexed.groupByKey().mapValues(compute_batch)

    all_candidates = batches.flatMap(lambda x: x[1]) \
        .map(lambda x: (min(x[0], x[1]), max(x[0], x[1]), x[2])) \
        .distinct() \
        .takeOrdered(TOP_N, key=lambda x: -x[2])  # top 100 unique pairs

    print("\nTop 100 similar pairs:")
    for uid, oid, sim in all_candidates:
        set1 = user_movies.lookup(uid)[0]
        set2 = user_movies.lookup(oid)[0]
        print(f"User {uid} & User {oid} -> Similarity: {sim:.3f}, Common Movies: {len(set1 & set2)}, Total Movies: {len(set1 | set2)}")

    result_df = spark.createDataFrame(all_candidates, ["user1", "user2", "similarity"])
    output_path = "hdfs:///user/rg4930_nyu_edu/movie_twins_output_large"
    result_df.write.mode("overwrite").csv(output_path, header=True)

    print(f"\nDone in {time.time() - start:.2f} seconds")
    spark.stop()

