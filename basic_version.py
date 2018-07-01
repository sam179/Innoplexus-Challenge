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