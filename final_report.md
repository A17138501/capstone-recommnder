### Movie Rating
github link: https://github.com/nyu-big-data/capstone-bdcs-79
contributions: question 1, 5: Rundong Guo; Question 2, 4: Jinran Jin; question 3: Nicole Lee, Qianqian Zhao. We all worked on the final report.
##### List of top 100 most similar pairs
This is saved to question1_large_result.csv on repo and is also included at the end of the report

##### A comparison between the average pairwise correlations between these highly similar pair and randomly picked pairs
To support customer segmentation through behavioral similarity, we developed a scalable PySpark pipeline to identify the top 100 most similar user pairs termed “movie twins” based solely on the sets of movies they rated, ignoring rating values. Using the datasketch library, we applied a MinHash-based locality-sensitive hashing (LSH) algorithm to approximate Jaccard similarity efficiently. To handle the full MovieLens dataset with over 330,000 users, we adopted a batching strategy, dividing user MinHash signatures into groups of 10,000 for memory-efficient processing. Within each batch, an LSH index was built to identify candidate pairs, and exact Jaccard similarities were computed only for queried matches. We retained only pairs with at least 20 rated movies per user and eliminated redundant comparisons. After aggregating all batches, we extracted the top 100 pairs with the highest similarity, ranging from 1.0 to 0.9102. This approach allowed us to scale to large data volumes while preserving accuracy, validating the use of MinHash-LSH for efficient and effective customer segmentation. 
I  list top 100 most similar pairs (including a suitable estimate of their similarity for each pair), sorted by similarity in question1_large_result.csv.
Our validation process applied Pearson correlation analysis to assess whether algorithmically identified movie twins demonstrated truly similar rating patterns. Using PySpark's distributed computing framework, we processed the full MovieLens dataset in memory-efficient batches, calculating correlations between users who had rated at least five movies in common. The diagnostic results showed compelling evidence of our algorithm's effectiveness: top pairs exhibited an average correlation of 0.1966 compared to just 0.0644 for random pairs, representing a difference of 0.1322. The top pairs achieved a maximum correlation of 0.9994, exceeding the random pairs' maximum of 0.9468, while both groups showed similar negative correlation floors (-0.3281 and -0.9540 respectively). Statistical validation through Welch's t-test produced a t-statistic of 2.8011 with a p-value of 0.0056, firmly establishing the statistical significance of this difference. With 99 valid correlations from our top pairs and 100 from random pairs, we had sufficient data to conclude that our minHash-based similarity identification successfully captured meaningful patterns in user preferences that transcended algorithmic artifacts, confirming the validity of our customer segmentation approach.

##### How the train test validation split is generated
To ensure reliable model training and evaluation, we partitioned the full MovieLens ratings dataset using a random split strategy via partition_ratings.py, dividing the dataset into three distinct subsets: training (60%), validation (20%), and test (20%). This separation enables us to perform fair model selection and final evaluation while avoiding data leakage. The resulting partitions are saved back to HDFS in CSV format with headers, making them compatible with downstream Spark pipelines for both baseline and ALS model evaluation. After execution, we verified that the files were successfully saved to the specified HDFS paths and that the row counts approximately reflected the 60/20/20 proportions. These partitioned datasets were used for all subsequent model training, tuning, and evaluation.

##### Popularity baseline
Our baseline recommendation model implemented a straightforward popularity-based approach, recommending the same top 100 movies to all users based solely on how many ratings each movie received in the training dataset. Using PySpark's distributed computing capabilities, we processed the MovieLens dataset efficiently, first identifying the most-rated movies (with IDs 318, 356, 296, 2571, and 593 leading the list) and then evaluating this one-size-fits-all approach against user preferences in the validation set. Performance metrics revealed modest effectiveness: precisionAt10 of 0.0644, ndcgAt10 of 0.0833, and mapAt10 of 0.0378, with slightly lower precision but higher ndcg when expanded to 100 recommendations (0.0366 and 0.1404 respectively). The sample evaluation data illustrated the inherent limitations of this approach—while some users had watched popular films like movie #318, many had unique preferences not captured by popularity alone, explaining the relatively low precision scores. This baseline establishes a performance floor against which our more sophisticated ALS-based collaborative filtering model will be compared, while confirming that even a simple model can capture some user preferences through the universal appeal of widely-watched films.

##### ALS Model training and validation
After finishing the baseline model, we moved on to building a personalized recommendation system using Spark’s ALS (alternating least squares) algorithm. The goal was to uncover hidden patterns between users and items by learning latent factors. We used the pyspark.ml.recommendation.ALS module and focused on tuning two key hyperparameters: the rank (which controls how many latent factors we learn) and the regularization parameter (which helps prevent overfitting). To make everything consistent, we used the same training, validation, and test splits we created earlier.We ran a grid search over several combinations of rank values {5, 10, 20} and regParam values {0.01, 0.1, 1.0}. Each model was trained with coldStartStrategy='drop' so we wouldn’t get undefined metrics, and we evaluated performance on the validation set using Spark’s built-in ranking metrics. The best model turned out to be the one with rank 20 and regParam 0.1, which got a precision@100 of 1.93e-5 and ndcg@100 of 1.70e-4. When compared to the baseline’s precision@100 of 0.0366 and ndcg@100 of 0.1404, the ALS model didn’t do very well. We took that best ALS config and retrained it on the full training data, then tested it. The test set results were slightly better, with precision@100 of 4.87e-5 and ndcg@100 of 4.04e-4. We saved these final results in als_final_eval_clean.csv, and all the validation results from tuning are in als_hyperparam_clean.csv for transparency and reproducibility.


user1,user2,similarity
88344,170112,1.0
127960,195168,1.0
13012,56180,1.0
313668,325068,1.0
51160,62784,1.0
14186,249611,1.0
169493,313668,1.0
169493,325068,1.0
186581,261572,1.0
92534,234310,1.0
9706,47690,1.0
47942,77318,1.0
47942,162798,1.0
77318,162798,1.0
184938,299674,1.0
12125,97685,1.0
59869,63309,1.0
23527,313463,1.0
81265,291625,1.0
141201,222025,1.0
125829,163485,1.0
123530,161017,1.0
168141,286014,0.984375
77647,86967,0.98046875
39826,263971,0.98046875
13918,71302,0.97265625
137715,138579,0.96875
248461,256733,0.96875
23867,87155,0.96875
260734,262614,0.96875
248461,292389,0.96484375
17906,80058,0.96484375
181762,248242,0.96484375
187157,266708,0.9609375
250852,266708,0.9609375
99030,115422,0.9609375
13195,49651,0.95703125
124266,159313,0.95703125
267144,305904,0.953125
181955,183123,0.94921875
10783,40479,0.94921875
136037,198214,0.94921875
49138,67802,0.94921875
59869,117141,0.9453125
63309,117141,0.9453125
77460,177837,0.9453125
92372,245387,0.9453125
226638,230806,0.9453125
115108,132844,0.94140625
61465,291625,0.9375
61465,81265,0.9375
77647,174815,0.9375
119572,172316,0.9375
298366,309198,0.9375
256733,292389,0.93359375
89343,95175,0.93359375
23867,38643,0.93359375
153073,228017,0.93359375
134981,277862,0.93359375
227884,236540,0.93359375
58500,234132,0.93359375
234310,309198,0.93359375
92534,309198,0.93359375
248242,308490,0.93359375
85653,124541,0.9296875
164677,292350,0.9296875
266708,293036,0.9296875
161970,187194,0.9296875
147443,202171,0.92578125
220645,248461,0.92578125
249077,266741,0.92578125
86856,88344,0.92578125
86856,170112,0.92578125
285976,324016,0.92578125
68284,257340,0.92578125
49138,263035,0.92578125
248242,313610,0.92578125
187157,250852,0.921875
29052,198820,0.921875
68852,306628,0.921875
19288,233568,0.921875
219326,230134,0.921875
202698,248242,0.921875
81745,290377,0.91796875
86967,174815,0.91796875
134061,263838,0.91796875
144788,169860,0.91796875
82321,273017,0.9140625
81527,194527,0.9140625
13195,26771,0.9140625
140841,242505,0.9140625
134061,168141,0.9140625
70628,257340,0.9140625
147898,159313,0.9140625
87326,230134,0.9140625
308490,313610,0.9140625
198381,293036,0.91015625
81265,83065,0.91015625
83065,291625,0.91015625
261129,272313,0.91015625
