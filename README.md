# sentiment_analysis

imdb_reviews.py is the code which includes 
-n_gram, 
-min_df, 
-tdidf, 
Whitelist is been added to remove unnecessary words which are not there in the stopwords..

p1.py does not include all of this.

This two are created and are being compared with respect to their accuracy..
With p1.py, the accuracy was 52.63%
and with imdb_reviews the accuracy was 79.02% using SVM and 78.47% using Xgboost.

I will use grid_search to improve the accuracy of imdb_reviews.It has not been added till now but will be added soon.
.
