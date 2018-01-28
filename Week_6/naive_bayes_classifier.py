#!/usr/local/bin/python3
import os
from math import log10
import pickle

DATA_DIR = os.path.dirname(__file__) + "/../datasets"
TRAIN_FILE = "r8-train-stemmed.txt"
TEST_FILE = "r8-test-stemmed.txt"
SAVE_FILE = "trained_model_naive_bayes.pkl"
PREDICTION_FILE = "predicted.txt"
RESULTS_FILE = "results.txt"

DELIM = '\t'

train_file_lines = open(os.path.join(DATA_DIR, TRAIN_FILE)).readlines()
test_file_lines = open(os.path.join(DATA_DIR, TEST_FILE)).readlines()

"""
Training the classifier
a.k.a. setting the probability values for each word
"""

# concatenated text for each document class
single_class_concat_dict = {}
# create a mega-doc from all the docs (helpful in calculating total counts of a word)
all_training_doc_concat_string = ' '.join([line.split(DELIM)[1] for line in train_file_lines])
# set of words in the mega-doc
vocab_set = set(all_training_doc_concat_string.split())
vocab_size = len(vocab_set)
# prior probability of document occuring in the training set P(Ci)
p_class_dict = {}
# likelihood of word occuring in a class
p_word_per_class_dict = {}

def train():
    print("Training: ")
    # set count of documents occuring in the training set n(Ci)
    n_classes = 0
    # count of each document class
    n_class_dict = {}
    for line in train_file_lines:
        doc_class, doc_body = line.split(DELIM)
        if doc_class not in n_class_dict:
            n_class_dict[doc_class] = 0
            single_class_concat_dict[doc_class] = ''
        # increase count for doc_class
        n_class_dict[doc_class] +=  1
        # append text to doc_class string
        single_class_concat_dict[doc_class] += ' ' + doc_body
        # increase totol classes seen count
        n_classes += 1

    for doc_class in n_class_dict:
        p_class_dict[doc_class] = (n_class_dict[doc_class]*1.0)/n_classes

    for word in vocab_set:
        p_word_per_class_dict[word] = {}
        for doc_class in n_class_dict:
            p_word_per_class_dict[word][doc_class] = (single_class_concat_dict[doc_class].count(word) + 1)/(len(single_class_concat_dict[doc_class].split()) + vocab_size + 1) # add 1-smoothing


try:
    with open(SAVE_FILE, 'rb') as input:
        p_class_dict, p_word_per_class_dict, single_class_concat_dict = pickle.load(input)
except:
    train()
    # Save the model
    with open(SAVE_FILE, 'wb') as output:
        pickle.dump([p_class_dict, p_word_per_class_dict, single_class_concat_dict], output, pickle.HIGHEST_PROTOCOL)

"""
Classification of test set
"""

predictions_file = open(PREDICTION_FILE,'w')
results_file = open(RESULTS_FILE, 'w')

print("Classification in progress: ")
correct_classifications = 0

for i in range(len(test_file_lines)):

    line = test_file_lines[i]
    print("Doc: "+str(i+1))
    actual_class_test = line.split(DELIM)[0]
    test_doc_body = line.split(DELIM)[1]

    score_per_class_dict = {}

    for doc_class in p_class_dict:
        # set current score as zero
        score_per_class_dict[doc_class] = log10(p_class_dict[doc_class])

        for word in test_doc_body.split():
            # take logarithm of the probability to avoid underflow
            if word not in p_word_per_class_dict:
                score_per_class_dict[doc_class] += log10(1/(len(single_class_concat_dict[doc_class].split()) + vocab_size + 1))
            else:
                score_per_class_dict[doc_class] += log10((single_class_concat_dict[doc_class].count(word) + 1)/(len(single_class_concat_dict[doc_class].split()) + vocab_size + 1))

    predicted_class_test = max(score_per_class_dict, key=score_per_class_dict.get)
    correct_classifications += (actual_class_test == predicted_class_test)
    predictions_file.write(','.join([str(i),str(actual_class_test), str(predicted_class_test), str(correct_classifications), "\n"]))

    print("Actual class: " + actual_class_test)
    print("Predicted class: " + predicted_class_test)
    print()

accuracy = (correct_classifications*1.0)/len(test_file_lines)
results_file.write("Accuracy: "+str(accuracy))

print("Accuracy: " + accuracy)
