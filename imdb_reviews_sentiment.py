# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from google.colab import files
uploaded = files.upload()

dataset = pd.read_csv('IMDB Dataset.csv',  encoding = "ISO-8859-1")

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer

sa_stop_words = nltk.corpus.stopwords.words("english")
    
#words that might invert a sentence's meaning
white_list = [
         'what', 'but', 'if', 'because', 'as', 'until', 'against',
        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'why', 'how', 'all', 'any',
        'most', 'other', 'some', 'such', 'no', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'can', 'will', 'just', 'dont', 'should']
    
#take these cut of the standard nltk stop word lists
sa_stop_words = [sw for sw in sa_stop_words if sw not in white_list]

corpus = []
for i in range(0, 50000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['review'][i])
    review = re.sub('<[^<]+?>', ' ', dataset['review'][i])
    tokenizer = RegexpTokenizer(r'\w+')
    review = tokenizer.tokenize(review)
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
#vectorize means we turn non_numeric data into an array of numbers
cv = CountVectorizer(
          max_features = 300,
          lowercase = True, #for demonstration true by default
          tokenizer = nltk.word_tokenize,#use the nltk tokenizer
          stop_words = sa_stop_words, #remove stop words
          min_df = 5, #minimum document frequency the word must appear more
         ngram_range = (1, 2)
    )

nltk.download('punkt')

X = cv.fit_transform(corpus)
from sklearn.feature_extraction import text
X = text.TfidfTransformer().fit_transform(X)

X = X.toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 42)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm
