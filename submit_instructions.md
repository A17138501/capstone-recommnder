### Q3_1 (partition_ratings.py)
spark-submit \
  --master yarn \
  --deploy-mode client \
  partition_ratings.py \
  --ratings_path hdfs:///user/xl5733_nyu_edu/ratings.csv \
  --train_path hdfs:///user/xl5733_nyu_edu/ml-latest-split/train \
  --val_path hdfs:///user/xl5733_nyu_edu/ml-latest-split/val \
  --test_path hdfs:///user/xl5733_nyu_edu/ml-latest-split/test

  spark-submit \
  --master yarn \
  --deploy-mode client \
  partition_ratings.py \
  --ratings_path hdfs:///user/jj3007_nyu_edu/capstone-bdcs-79/ml-latest/ratings.csv \
  --train_path hdfs:///user/jj3007_nyu_edu/capstone-bdcs-79/ml-latest-split/train \
  --val_path hdfs:///user/jj3007_nyu_edu/capstone-bdcs-79/ml-latest-split/val \
  --test_path hdfs:///user/jj3007_nyu_edu/capstone-bdcs-79/ml-latest-split/test

