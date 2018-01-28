#!/usr/local/bin/python3
from bisect import bisect_left
from math import log


"""Returns data and its label (X,y)"""


def get_data_and_label(raw_data, delim='\t', label_pos=0, data_pos=1):
    labels = [line.split(delim)[label_pos] for line in raw_data]
    data = [line.split(delim)[data_pos] for line in raw_data]
    return (data, labels)


"""Returns sorted list of unique words in a list of documents (strings)."""


def get_vocabulary(documents):
    return sorted(list(set(' '.join(documents).split())))  # sorted list of unique words


"""return wf (word frequency) vector space for a list of documents"""


def bag_of_words(documents):
    v_space = []
    vocab = get_vocabulary(documents)
    for doc in documents:
        feature_vector = [0 for word in vocab]
        for word in doc.split():
            pos = bisect_left(vocab, word)
            if pos < len(vocab):
                feature_vector[pos] += 1
        v_space.append(feature_vector)
    return v_space


"""returns tf scaled by the idf"""


def tf_idf(documents):
    vocab = get_vocabulary(documents)
    v_space = bag_of_words(documents)
    for i in range(len(vocab)):
        doc_count = 0
        for j in range(len(documents)):
            if v_space[j][i] > 0:
                doc_count += 1  # increment doc_count
                v_space[j][i] /= (1.0 * len(documents[j].split()))
        # """remove this"""
        # print(v_space)
        # print(doc_count)
        for j in range(len(documents)):
            v_space[j][i] *= log(1 + len(documents) / (1.0 * (doc_count)))
    return v_space
