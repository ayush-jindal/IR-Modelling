import math
import json
import string

from corpus_creation import vector_space_model_Kgrams_pos_index

import numpy as np
			
def log_tf_and_idf(term, query, corpus, N):

	'''
		finds log tf of a given term for all the docs
		if corpus passed, idf will also be calculated
	'''

	log_tf_idf_q = 0
	log_tf_d = {}

	if term in corpus:
		# the first zero is the default docID for each query term
		# the next is to get the frequency
		log_tf_idf_q = (1 + math.log10(query[term][0][0])) * math.log10(N/len(corpus[term]))
		for doc in corpus[term]:
			log_tf_d[doc] = 1 + math.log10(corpus[term][doc][0])
	
	return log_tf_idf_q, log_tf_d
		
def cos_norm(vector):

	'''
		vector: dict with term as keys as their scores as respective values
		returns a modifed dict after normalizing it document wise
	'''
	
	# vector will be a dict with term as keys
	mag = np.linalg.norm(vector)
	
	# normalizes by dividing by the magnitude
	if mag != 0:
		vector /= mag
	return vector
	
def calc_dot_product(query, doc):
	'''
		query: dict
		doc: dict
		
		computes the final score by multiplying the weights of the query terms with their document counterpart
	'''

	return np.array(query[:len(doc)]).dot(np.array(doc))
	
def calc_score(term, terms, docs, N):

	query_vector = []
	doc_vectors = {}
	doc_scores = {}
	
	# performs lt/ln weight computation on query
	query_val, doc_vecs = log_tf_and_idf(term, terms, docs, N)
	if query_val:
		query_vector.append(query_val)
		for doc in doc_vecs:
			if doc in doc_vectors:
				doc_vectors[doc].append(doc_vecs[doc])
			else:
				doc_vectors[doc] = [doc_vecs[doc]]
	
	# performs c computation on query
	#query_vector = cos_norm(query_vector)

	if not len(query_vector):
		print('No such determinant term inside the database')
		return {}
	
	# traverses through each docID in the dict and computes lnc which gets stored in doc_vectors dict against each docID
	for doc in doc_vectors:
		#doc_vectors[doc] = cos_norm(doc_vectors[doc])

		# calculates the dot product between query and the current doc
		doc_scores[doc] = calc_dot_product(query_vector, doc_vectors[doc])

	return doc_scores

def calc_score_posn(start, terms, docs):

	doc_scores = {}

	for i, term1 in enumerate(tuple(terms.keys())[start:-1], start=start):		
		for j, term2 in enumerate(tuple(terms.keys())[i+1:], start=i+1):
			if term2 in docs:
				opt_dist = j - i
				common_docs = set(docs[term1]).intersection(set(docs[term2]))
				for doc in common_docs:
					score = 0
					locs1 = np.array(docs[term1][doc][1])
					locs2 = np.array(docs[term2][doc][1])
					for loc1 in locs1:
						rel_dist = abs((locs2 - loc1)/opt_dist)
						score += sum([x if x<=1 else 1/x for x in rel_dist])

					if doc in doc_scores:
						doc_scores[doc] += score
					else:
						doc_scores[doc] = score

	return doc_scores

def okapi_type(query, corpus, meta, best, K, stopwords_file):
	stopwords = []
	if stopwords_file:		
		with open(stopwords_file, 'r') as file:
			stopwords = json.load(file)

	terms_Kgram = {}
	vector_space_model_Kgrams_pos_index(query, terms_Kgram, K, stopwords=stopwords)
	corpus_Kgram = corpus.format(K)
	# loads the corpus and title files into the two dicts
	with open(corpus_Kgram, 'r') as file:
		docs_Kgram = json.load(file)

	if K != 1:
		terms_1gram = {}
		vector_space_model_Kgrams_pos_index(query, terms_1gram, 1, stopwords=stopwords)
		corpus_1gram = corpus.format(1)
		# loads the corpus and title files into the two dicts
		with open(corpus_1gram, 'r') as file:
			docs_1gram = json.load(file)
	else:
		terms_1gram, docs_1gram = terms_Kgram, docs_Kgram

	with open(meta, 'r') as file:
		doc_meta = json.load(file)

	N = len(doc_meta)
	avg_len = sum([doc_meta[doc][1] for doc in doc_meta])/N
	k1 = 1.6
	b = 0.75

	final_scores = {}
	#final_scores_posn = dict.fromkeys(doc_meta.keys(), 1)
	#final_scores_vector = {}

	for i, term in enumerate(terms_Kgram):
		score_vector = calc_score(term, terms_Kgram, docs_Kgram, N)
		
		if len(terms_1gram) > 1:
			score_posn = calc_score_posn(i, terms_1gram, docs_1gram)
			all_docs = set(score_vector).union(set(score_posn))

			for doc in all_docs:
				#final_scores[doc] = final_scores.get(doc, 0) + score_vector.get(doc, 0) + 5*score_posn.get(doc, 0)
				#final_scores_posn[doc] = final_scores_posn.get(doc, 0) + score_posn.get(doc, 0)
				#final_scores_vector[doc] = final_scores_vector.get(doc, 0) + score_vector.get(doc, 0)
				# these 1+ done to avoid breaking of avoid if one of the terms zero
				final_scores[doc] = final_scores.get(doc, 0) + (1 + score_vector.get(doc, 0)) * (k1 + 1)*(1 + score_posn.get(doc, 0))/(1 + score_posn.get(doc, 0) + k1*((1-b) + b*doc_meta[doc][1]/avg_len))
				
		else:
			for doc in score_vector:
				final_scores[doc] = final_scores.get(doc, 0) + score_vector.get(doc, 0)
			
	final_scores = {k: v for k, v in sorted(final_scores.items(), key=lambda item: item[1], reverse=True)}
	#final_scores_posn = {k: v for k, v in sorted(final_scores_posn.items(), key=lambda item: item[1], reverse=True)}
	#final_scores_vector = {k: v for k, v in sorted(final_scores_vector.items(), key=lambda item: item[1], reverse=True)}
	#print(sorted_scores)
	# picks top K documents
	temp = best
	for doc in final_scores:
		if temp > 0:
			print(doc_meta[doc][0], final_scores[doc])
		else:
			break
		temp -= 1
	print('*'*20)
	'''temp = best
	for doc in final_scores_posn:
		if temp > 0:
			print(doc_meta[doc][0], final_scores_posn[doc])
		else:
			break
		temp -= 1
	print('*'*20)
	for doc in final_scores_vector:
		if best > 0:
			print(doc_meta[doc][0], final_scores_vector[doc])
		else:
			break
		best -= 1'''

def search(corpus, meta, query, best, K, stopwords_file):
	okapi_type(query, corpus, meta, best, K, stopwords_file)
