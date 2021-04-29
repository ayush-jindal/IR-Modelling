import re
import os
import json
import string
			
def vector_space_model_Kgrams_pos_index(lines, docs, K, doc_id=0, stopwords=None):

	words = re.sub('[^a-zA-Z0-9]', ' ', lines).split(' ')

	if stopwords:
		words = [word.lower() for word in words if word not in stopwords]
	else:
		words = [word.lower() for word in words if word != '']
	
	for pos in range(len(words) - K + 1):
		word = ' '.join(words[pos: pos+K])

		if word in docs:
			if doc_id in docs[word]:
				docs[word][doc_id][0] += 1
				docs[word][doc_id][1].append(pos)
			else:
				docs[word][doc_id] = [1, [pos]]
		else:
			docs[word] = {}
			docs[word][doc_id] = [1, [pos]]
			
	return len(words)

def corpus_creation(filename, corpus, meta, data, K, stopwords_file):
	'''
		filename: str the abs/rel path of the raw corpus file
		
		creates the vector space model out of it to save in two file
			1) filename.json: the main corpus
			2) filename_titles.json: contains the doc titles with their ID
	'''
	
	save_name = filename.split('/')[-1].split('.')[0]
	corpus = corpus.format(K)

	if os.path.isfile(data):
		with open(data, 'r') as file:
			files = file.readlines()
			if save_name+'_'+str(K)+('_no_sw' if stopwords_file else '')+'\n' in files:
				print('This file already in corpus')
				return

	with open(data, 'a+') as file:
		print(save_name+'_'+str(K)+('_no_sw' if stopwords_file else ''), file=file)
	
	# reading in the file
	with open(filename, 'r') as file:
		lines = file.readlines()

	stopwords = []
	if stopwords_file:		
		with open(stopwords_file, 'r') as file:
			stopwords = json.load(file)

	# to remove opening <>
	exp_open = '<.*?>'
	# to remove closing </>
	exp_close = '</.>'
	# to extract values from the <doc>
	exp_id = '".*?"'
	
	# the two main dicts
	if os.path.isfile(corpus):
		with open(corpus, 'r') as file:
			docs = json.load(file)
	else:
		docs = {}
	if os.path.isfile(meta):
		with open(meta, 'r') as file:
			doc_data = json.load(file)
	else:
		doc_data = {}
	
	for i, line in enumerate(lines):
		# extracting id, title from <doc> tag and creating a new doc dict for this doc
		if '<doc' in line:
			tags = re.findall(exp_id, line)
			doc_id = int(tags[0][1:-1])
			doc_title = tags[-1][1:-1]
			body = ''
			continue
		# adding the created doc dict to the main docs dict
		if '/doc>' in line:
			#vector_space_model_Kgrams(lines, doc, stopwords)
			length = vector_space_model_Kgrams_pos_index(body, docs, K, doc_id, stopwords=stopwords)
			doc_data[doc_id] = [doc_title, length]
			continue
		# removing non-useful tags and creating vector space model from the current line
		line = re.sub(exp_open, '', line)
		line = re.sub(exp_close, '', line)
		body += line
	
	# saving the two dicts as json files
	
	with open(corpus, 'w') as file:
		json.dump(docs, file)
	with open(meta, 'w') as file:
		json.dump(doc_data, file)
