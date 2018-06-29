# Innoplexus hackathon

# Importing libraries

import os 
import re
import nltk
import logging
import numpy as np
import logging
import pandas as pd 
import nltk.data 
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from xgboost import XGBClassifier
from gensim.models import word2vec

#  Importing data from files

def import_data():

	training_info = pd.read_csv('information_train.csv', header = 0, delimiter = '\t')
	testing_info = pd.read_csv('information_test.csv', header = 0, delimiter = '\t')
	train_data = pd.read_csv('train.csv', header = 0, delimiter = '\t')
	test_data = pd.read_csv('test.csv', header = 0, delimiter = '\t')

	return training_info, testing_info, train_data, test_data

#  Creating maps

def create_maps(data):

	training_info = data[0]
	testing_info = data[1]
	text_map = pd.concat([training_info[['pmid', 'abstract', 'full_Text']], testing_info['pmid', 'abstract', 'full_Text']])
	title_map = pd.concat([training_info[['pmid', 'article_title']], testing_info[['pmid', 'article_title']]])	
	set_map = pd.concat([training_info[['pmid', 'set']], testing_info[['pmid', 'set']]])


#  Function parses a sentence and returns a list of words

def sentence_parser(sentence_data):

	sentence_data = BeautifulSoup(sentence_data).get_text()
	sentence_data = re.sub("[^a-zA-Z]"," ", sentence_data)
	word_list = sentence_data.lower().split()
	return word_list

#  Function parses raw text and converts them to sentences. This is done using 'tokenizer' library of nltk. We obtain a list of sentences.
#  Finally, we obtain a list of lists of words. 	

def text_parser(text_data):

	nltk.download()
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	raw_sentences = tokenizer.tokenize(review.strip())
	sentences = []

	for raw_sentence in raw_sentences:
		if len(raw_sentence) > 0:
			sentences.append(sentence_parser(raw_sentence))

	return sentences

#  This handles solely the text part

def text_proccessing(text_data):

	import logging
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)

	sentences = []

	for text in text_data['abstract']:
		sentences += text_parser(text)

	for text in test_data['full_Text']:
		sentences += text_parser(text)

	model = word2vec.Word2Vec(sentences, workers = 4, min_count = 40, size = 300, window = 10, sample = 1e-3)
	model.init_sims(replace = True)
	model_name = "word2vec_representations"
	model.save(model_name)


def main():

	data = import_data()
	create_maps(data)

if __name__ == "__main__":
	main()

'''
1. Import data. 
2. convert author_name to index, using small dictionary
3. date to age conversion
4. 
'''