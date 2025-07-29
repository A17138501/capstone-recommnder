from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import collect_list, col, explode
from pyspark.sql import Row

if __name__ == "__main__":
    spark = SparkSession.builder.appName("ALS Grid Search").getOrCreate()

    train_path = "/user/rg4930_nyu_edu/partitioned_ratings_parquet/train"
    val_path = "/user/rg4930_nyu_edu/partitioned_ratings_parquet/val"
    test_path = "/user/rg4930_nyu_edu/partitioned_ratings_parquet/test"

    train_df = spark.read.parquet(train_path)
    val_df = spark.read.parquet(val_path)
    test_df = spark.read.parquet(test_path)

    results = []

    for rank in [5, 10, 20]:
        for reg in [0.01, 0.1, 1.0]:
            print(f"Training ALS model with rank={rank}, regParam={reg}")
            als = ALS(
                userCol="userId",
                itemCol="movieId",
                ratingCol="rating",
                rank=rank,
                regParam=reg,
                maxIter=5,
                coldStartStrategy="drop",
                nonnegative=True
            )

            model = als.fit(train_df)
            recs = model.recommendForAllUsers(100)

            preds = recs.select("userId", explode("recommendations").alias("rec"))
            preds = preds.select("userId", col("rec.movieId").alias("movieId"))
            pred_topk = preds.groupBy("userId").agg(collect_list("movieId").alias("predicted_items"))

            truth = val_df.groupBy("userId").agg(collect_list("movieId").alias("true_items"))
            joined = pred_topk.join(truth, on="userId", how="inner")
            ranking_rdd = joined.select("predicted_items", "true_items").rdd.map(tuple)

            metrics = RankingMetrics(ranking_rdd)
            p100 = metrics.precisionAt(100)
            ndcg = metrics.ndcgAt(100)

            results.append(Row(model="als", rank=int(rank), regParam=float(reg),
                               precision=float(p100), ndcg=float(ndcg)))
            print(f"rank={rank}, regParam={reg}, precision@100={p100}, ndcg@100={ndcg}")

    spark_results = spark.createDataFrame(results)
    spark_results.write.mode("overwrite").option("header", True).csv("als_hyperparam_log")

    # Step 2: Select best param by precision
    best_row = max(results, key=lambda x: x.precision)
    best_rank = best_row.rank
    best_reg = best_row.regParam
    print(f"\nBest hyperparams from validation: rank={best_rank}, regParam={best_reg}")

    # Step 3: Retrain best model and evaluate on test
    print("\nEvaluating best ALS model on test set...")

    best_als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=best_rank,
        regParam=best_reg,
        maxIter=10,
        coldStartStrategy="drop",
        nonnegative=True
    )

    final_model = best_als.fit(train_df)
    test_recs = final_model.recommendForAllUsers(100)

    preds = test_recs.select("userId", explode("recommendations").alias("rec"))
    preds = preds.select("userId", col("rec.movieId").alias("movieId"))
    pred_topk = preds.groupBy("userId").agg(collect_list("movieId").alias("predicted_items"))

    truth = test_df.groupBy("userId").agg(collect_list("movieId").alias("true_items"))
    joined = pred_topk.join(truth, on="userId", how="inner")
    ranking_rdd = joined.select("predicted_items", "true_items").rdd.map(tuple)

    metrics = RankingMetrics(ranking_rdd)
    p100 = metrics.precisionAt(100)
    ndcg = metrics.ndcgAt(100)

    print(f"\nFinal Test Set Evaluation:")
    print(f"precision@100: {p100:.4f}")
    print(f"ndcg@100: {ndcg:.4f}")

    final_result = spark.createDataFrame([
        Row(model="als_final", rank=best_rank, regParam=best_reg,
            precision=float(p100), ndcg=float(ndcg))
    ])
    final_result.write.mode("overwrite").option("header", True).csv("als_final_eval_log")

