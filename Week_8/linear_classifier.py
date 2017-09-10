#!/usr/local/bin/python3
import os
import pickle
from prepare_data import *
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

FILE_DIR = os.path.dirname(__file__) + "./datasets"
TRAIN_FILE = "r8-train-stemmed.txt"
TEST_FILE = "r8-test-stemmed.txt"
SAVE_FILE = "trained_model_linear.pkl"
PREDICTION_FILE = "predicted_linear.txt"
RESULTS_FILE = "results_linear.txt"

DELIM = '\t'

train_file_lines = open(os.path.join(FILE_DIR, TRAIN_FILE)).readlines()
test_file_lines = open(os.path.join(FILE_DIR, TEST_FILE)).readlines()

training_documents, training_labels = get_data_and_label(train_file_lines)
test_documents, test_labels = get_data_and_label(test_file_lines)

vocabulary = get_vocabulary(training_documents)
classes = get_vocabulary(training_labels)
class_num_dict = {}


print("Preparing data: ")
# assign a number to each class
for i in range(len(classes)):
	class_num_dict[classes[i]] = i+1

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(training_documents)
y_train = np.array(training_labels)
X_test = vectorizer.transform(test_documents)
y_test = np.array(test_labels)


print("Training: ")
classifier = LogisticRegression(multi_class="multinomial", solver="lbfgs")
classifier.fit(X_train, y_train)


print("Testing: ")
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("Accuracy score: " + str(score))
