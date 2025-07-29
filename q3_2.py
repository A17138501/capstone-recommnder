{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part 3\n",
    "\n",
    "As a first step, you will need to partition the ratings data into training, validation, and test sets. We recommend writing a script do this in advance, and saving the partitioned data for future use. This will reduce the complexity of your code down the line, and make it easier to generate alternative splits if you want to assess the stability of your implementation.\n",
    "\n",
    "### Partitioning MovieLens Ratings Data for Recommendation System"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    }
   ],
   "source": [
    "ratings_large = spark.read.csv(\"ml-latest/ratings.csv\", header=True, inferSchema=True)\n",
    "ratings_small = spark.read.csv(\"ml-latest-small/ratings.csv\", header=True, inferSchema=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyspark.sql.functions import col, rand\n",
    "from pyspark.sql.window import Window\n",
    "\n",
    "def partition_ratings_data(ratings_df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):\n",
    "    \n",
    "    # Validate ratios\n",
    "    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-8, \"Ratios must sum to 1\"\n",
    "    \n",
    "    # Create reproducible random split by user\n",
    "    window = Window.partitionBy(\"userId\").orderBy(\"timestamp\")\n",
    "    \n",
    "    partitioned = ratings_df.withColumn(\"row_num\", F.row_number().over(window)) \\\n",
    "                          .withColumn(\"user_count\", F.count(\"*\").over(Window.partitionBy(\"userId\"))) \\\n",
    "                          .withColumn(\"rand\", rand(seed))\n",
    "    \n",
    "    # For users with enough ratings, split by ratio\n",
    "    partitioned = partitioned.withColumn(\n",
    "        \"split\",\n",
    "        F.when(\n",
    "            col(\"user_count\") >= 10,  # Only split users with sufficient ratings\n",
    "            F.when(\n",
    "                col(\"rand\") < train_ratio, \"train\"\n",
    "            ).when(\n",
    "                col(\"rand\") < (train_ratio + val_ratio), \"val\"\n",
    "            ).otherwise(\"test\")\n",
    "        ).otherwise(\"train\")  # Small users go entirely to train set\n",
    "    )\n",
    "    \n",
    "    # Split the data\n",
    "    train_df = partitioned.filter(col(\"split\") == \"train\").drop(\"row_num\", \"user_count\", \"rand\", \"split\")\n",
    "    val_df = partitioned.filter(col(\"split\") == \"val\").drop(\"row_num\", \"user_count\", \"rand\", \"split\")\n",
    "    test_df = partitioned.filter(col(\"split\") == \"test\").drop(\"row_num\", \"user_count\", \"rand\", \"split\")\n",
    "    \n",
    "    return train_df, val_df, test_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Partitioning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training set: 60528 ratings\n",
      "Validation set: 20198 ratings\n",
      "Test set: 20110 ratings\n",
      "Total: 100836 ratings\n"
     ]
    }
   ],
   "source": [
    "# Partition the data \n",
    "train_ratings_small, val_ratings_small, test_ratings_small = partition_ratings_data(ratings_small)\n",
    "\n",
    "# Show distribution\n",
    "print(f\"Training set: {train_ratings_small.count()} ratings\")\n",
    "print(f\"Validation set: {val_ratings_small.count()} ratings\")\n",
    "print(f\"Test set: {test_ratings_small.count()} ratings\")\n",
    "print(f\"Total: {train_ratings_small.count() + val_ratings_small.count() + test_ratings_small.count()} ratings\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training set: 20391449 ratings\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation set: 6719354 ratings\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test set: 6721359 ratings\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Stage 90:========================================>            (151 + 16) / 200]\r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total: 33832162 ratings\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    }
   ],
   "source": [
    "# Partition the data\n",
    "train_ratings_large, val_ratings_large, test_ratings_large = partition_ratings_data(ratings_large)\n",
    "\n",
    "# Show distribution\n",
    "print(f\"Training set: {train_ratings_large.count()} ratings\")\n",
    "print(f\"Validation set: {val_ratings_large.count()} ratings\")\n",
    "print(f\"Test set: {test_ratings_large.count()} ratings\")\n",
    "print(f\"Total: {train_ratings_large.count() + val_ratings_large.count() + test_ratings_large.count()} ratings\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Save Partitioned Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def save_partitioned_data(train_df, val_df, test_df, base_path):\n",
    "    \"\"\"Save partitioned data to persistent storage\"\"\"\n",
    "    # Save as parquet files\n",
    "    train_df.write.parquet(f\"{base_path}/train\", mode=\"overwrite\")\n",
    "    val_df.write.parquet(f\"{base_path}/val\", mode=\"overwrite\")\n",
    "    test_df.write.parquet(f\"{base_path}/test\", mode=\"overwrite\")\n",
    "    \n",
    "    # Also save as CSV for easy inspection\n",
    "    train_df.coalesce(1).write.csv(\n",
    "        f\"{base_path}/train_csv\", \n",
    "        header=True, \n",
    "        mode=\"overwrite\"\n",
    "    )\n",
    "    val_df.coalesce(1).write.csv(\n",
    "        f\"{base_path}/val_csv\", \n",
    "        header=True, \n",
    "        mode=\"overwrite\"\n",
    "    )\n",
    "    test_df.coalesce(1).write.csv(\n",
    "        f\"{base_path}/test_csv\", \n",
    "        header=True, \n",
    "        mode=\"overwrite\"\n",
    "    )\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Save Partitioned Data "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save  small \n",
    "save_partitioned_data(train_ratings_small, val_ratings_small, test_ratings_small, \"ml-latest-small/partitions\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    }
   ],
   "source": [
    "# Save  large datasets\n",
    "save_partitioned_data(train_ratings_large, val_ratings_large, test_ratings_large, \"ml-latest/partitions\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "train: <a href='ml-latest-small/partitions/train_csv/part-00000-cf6c8cca-57c7-4477-9a11-a849201b4e1e-c000.csv' target='_blank'>ml-latest-small/partitions/train_csv/part-00000-cf6c8cca-57c7-4477-9a11-a849201b4e1e-c000.csv</a><br>"
      ],
      "text/plain": [
       "/home/qz2671/Customer_segmentation/ml-latest-small/partitions/train_csv/part-00000-cf6c8cca-57c7-4477-9a11-a849201b4e1e-c000.csv"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "val: <a href='ml-latest-small/partitions/val_csv/part-00000-c47ef52a-421c-48e0-8cdd-d8107f01d016-c000.csv' target='_blank'>ml-latest-small/partitions/val_csv/part-00000-c47ef52a-421c-48e0-8cdd-d8107f01d016-c000.csv</a><br>"
      ],
      "text/plain": [
       "/home/qz2671/Customer_segmentation/ml-latest-small/partitions/val_csv/part-00000-c47ef52a-421c-48e0-8cdd-d8107f01d016-c000.csv"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "test: <a href='ml-latest-small/partitions/test_csv/part-00000-476c3ec7-0841-42c0-aab6-cd28356ab36b-c000.csv' target='_blank'>ml-latest-small/partitions/test_csv/part-00000-476c3ec7-0841-42c0-aab6-cd28356ab36b-c000.csv</a><br>"
      ],
      "text/plain": [
       "/home/qz2671/Customer_segmentation/ml-latest-small/partitions/test_csv/part-00000-476c3ec7-0841-42c0-aab6-cd28356ab36b-c000.csv"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from IPython.display import FileLink\n",
    "import glob\n",
    "import os\n",
    "\n",
    "def create_download_links(base_path):\n",
    "    \"\"\"\n",
    "    Create download links for partition files in Jupyter Notebook\n",
    "    \n",
    "    Args:\n",
    "        base_path (str): Path where partitions were saved\n",
    "    \"\"\"\n",
    "    partitions = {\n",
    "        'train': f\"{base_path}/train_csv/part-00000-*.csv\",\n",
    "        'val': f\"{base_path}/val_csv/part-00000-*.csv\",\n",
    "        'test': f\"{base_path}/test_csv/part-00000-*.csv\"\n",
    "    }\n",
    "    \n",
    "    for name, pattern in partitions.items():\n",
    "        try:\n",
    "            file = glob.glob(pattern)[0]\n",
    "            display(FileLink(file, result_html_prefix=f\"{name}: \"))\n",
    "        except IndexError:\n",
    "            print(f\"File not found: {pattern}\")\n",
    "\n",
    "create_download_links(\"ml-latest-small/partitions\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "train: <a href='ml-latest/partitions/train_csv/part-00000-6ecab410-2ec2-45e5-8c6f-7aa0e3b55521-c000.csv' target='_blank'>ml-latest/partitions/train_csv/part-00000-6ecab410-2ec2-45e5-8c6f-7aa0e3b55521-c000.csv</a><br>"
      ],
      "text/plain": [
       "/home/qz2671/Customer_segmentation/ml-latest/partitions/train_csv/part-00000-6ecab410-2ec2-45e5-8c6f-7aa0e3b55521-c000.csv"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "val: <a href='ml-latest/partitions/val_csv/part-00000-6217eebb-2756-4dc9-baf7-5790d3afb255-c000.csv' target='_blank'>ml-latest/partitions/val_csv/part-00000-6217eebb-2756-4dc9-baf7-5790d3afb255-c000.csv</a><br>"
      ],
      "text/plain": [
       "/home/qz2671/Customer_segmentation/ml-latest/partitions/val_csv/part-00000-6217eebb-2756-4dc9-baf7-5790d3afb255-c000.csv"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "test: <a href='ml-latest/partitions/test_csv/part-00000-be8ab68a-3eb0-46b6-8735-5d0545746b82-c000.csv' target='_blank'>ml-latest/partitions/test_csv/part-00000-be8ab68a-3eb0-46b6-8735-5d0545746b82-c000.csv</a><br>"
      ],
      "text/plain": [
       "/home/qz2671/Customer_segmentation/ml-latest/partitions/test_csv/part-00000-be8ab68a-3eb0-46b6-8735-5d0545746b82-c000.csv"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "create_download_links(\"ml-latest/partitions\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
