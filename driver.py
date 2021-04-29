import sys
import getopt
import os

from corpus_creation import corpus_creation
from main import search

def main(argv):

	corpus = 'corpus'
	meta = 'meta'
	new_file = ''
	stopwords_file = ''
	K = 1
	best = 10
	file_data = 'files_created'
	
	try:
		opts, args = getopt.getopt(argv, 'hq:b:k:n:', ['help', 'query=', 'best=', 'stopwords=', 'new-file='])
	except getopt.GetoptError:
		print('error')
		sys.exit(1)
	for opt, arg in opts:
		if opt in ('-h', '--help'):
			print('-h help\n-c main corpus file\n-t file with document titles\n-r file whose corpus is to created\n-q query')
			sys.exit(0)
		elif opt in ('-q', '--query'):
			query = arg
		elif opt in ('-b', '--best'):
			best = int(arg)
		elif opt in ('--stopwords'):
			stopwords_file = arg
		elif opt in ('-k'):
			K = int(arg)
		elif opt in ('-n', '--new-file'):
			new_file = arg

	#corpus = corpus+'_{}-gram{}.json'.format('_no_sw' if stopwords_file else '')
	corpus = corpus+'_{}-gram{}.json'.format('{}', '_no_sw' if stopwords_file else '')
	meta = meta+'{}.json'.format('_no_sw' if stopwords_file else '')

	if new_file != '':
		corpus_creation(new_file, corpus, meta, file_data, K, stopwords_file)
		corpus_creation(new_file, corpus, meta, file_data, 1, stopwords_file)
	
	if query == '':
		query = input('Enter your query: ')

	if os.path.isfile(corpus.format(K)):
		search(corpus, meta, query, best, K, stopwords_file)
	else:
		print('Need to create a new corpus\nCan do so by creating using -n')
		sys.exit(-1)
		
if __name__ == '__main__':
	main(sys.argv[1:])
