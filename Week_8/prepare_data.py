#!/usr/local/bin/python3
from bisect import bisect_left
from math import log

def sample():
	return 1

def get_vocabulary(documents):
	return sorted(list(set(' '.join(documents).split()))) # sorted list of unique words


'''
	return wf (word frequency) vector space
'''
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
		feature_vector = []
	return v_space


'''
	returns bag of words scaled by the idf
'''
def tf_idf(documents):
	vocab = get_vocabulary(documents)
	v_space = bag_of_words(documents)
	for i in range(len(vocab)):
		doc_count = 0
		for j in range(len(documents)):
			if v_space[j][i] > 0:
				doc_count += 1 #increment doc_count
		"""remove this"""
		print(doc_count)
		for j in range(len(documents)):
			v_space[j][i] *= log(len(documents)/(1.0 * (doc_count+1)))
	return v_space
