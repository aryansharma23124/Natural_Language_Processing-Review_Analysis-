import numpy as np
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')
print("dataset imported")

import re

import nltk

nltk.download('stopwords')

# to remove stopword
from nltk.corpus import stopwords

# for Stemming propose
#from nltk.stem.porter import PorterStemmer
from porter import PorterStemmer
p=PorterStemmer()
p.stem("Alcoholic")

# Initialize empty array
# to append clean text
corpus = []
for i in range(0, 1000):
    # column : "Review", row ith
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # convert all cases to lower cases
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()

    # loop for stemming each word
    # in string array at ith row
    review = [ps.stem(word) for word in review
              if not word in set(stopwords.words('english'))]

    # rejoin all string array elements
    # to create back into a string
    review = ' '.join(review)

    # append each string to create
    # array of clean text
    corpus.append(review)
print("Completed")

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

# To extract max 1500 feature.
# "max_features" is attribute to
# experiment with to get better results
cv = CountVectorizer(max_features=1500)

# X contains corpus (dependent variable)
X = cv.fit_transform(corpus).toarray()

# y contains answers if review
# is positive or negative
y = dataset.iloc[:, 1].values
# Splitting the dataset into
# the Training set and Test set
from sklearn.model_selection import train_test_split

# experiment with "test_size"
# to get better results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
# Fitting Random Forest Classification
# to the Training set
from sklearn.ensemble import RandomForestClassifier

# n_estimators can be said as number of
# trees, experiment with n_estimators
# to get better results
model = RandomForestClassifier(n_estimators=501,
                               criterion='entropy')

model.fit(X_train, y_train)
# Predicting the Test set results
y_pred = model.predict(X_test)

print(y_pred.size)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("True Positive-Positive reviews correctly identified")
print(cm[0][0])
print("True Negative-Positive reviews  not correctly identified")
print(cm[0][1])
print("False Positive-Neative reviews correctly identified")
print(cm[1][0])
print("False Negative-Positive reviews correctly identified")
print(cm[1][1])


