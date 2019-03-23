import numpy as np
import pandas as pd
import string
import os
import sys
from time import time
from random import randint
import sys
import nltk
from nltk.tokenize import word_tokenize
import gensim
from util import *

lock_factor=1.0
window_size = 3
num_neg_samples = 10
num_epochs = 5
learning_rate = 0.001
min_count=1
punkt = set(list(string.punctuation)+["``","`"])

def get_corpus(dataset_dir):
	corpus = []
	for filename in os.listdir(dataset_dir):
		if filename.endswith(".txt"):
		    print(filename)
		    with open(os.path.join(dataset_dir,filename), 'r') as f:
		        lines = f.readlines()
		        lines = " ".join(lines)
		        lines = tokenize_doc(lines)
		        # print(len(lines))
		        corpus.append(lines)
	return corpus

def combine_vocab(model, domain_vocab):
	word2vec_vocab = list(model.wv.vocab.keys())
	domain_vocab = list(domain_vocab.keys())
	combined_vocab = word2vec_vocab
	for x in domain_vocab:
	    if x not in word2vec_vocab:
	        combined_vocab.append(x)
	return combined_vocab


def get_domain_model(corpus, word2vec_model):
	# check size of embedding of word2vec
	embedding_dim = word2vec_model.vectors[0].shape[0]
	domain_model = gensim.models.Word2Vec(size=300, alpha=0.025,
               window=5, min_count=2, max_vocab_size=None, 
               sample=0.001, workers=4, min_alpha=0.0001, 
               sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1)
	domain_model.build_vocab(corpus)
	total_examples = domain_model.corpus_count
	domain_model.build_vocab([list(word2vec_model.vocab.keys())], update=True)
	domain_model.intersect_word2vec_format(pretrained_embeddings_path, binary=True, lockf=lock_factor)
	domain_model.train(corpus, total_examples=total_examples, epochs=1)
	return domain_model

def tag_pronoun(tokens):
    tags = nltk.pos_tag(tokens)
    for i,(key,val) in enumerate(tags):
        if val=='NNP' or val=='NNPS':
            tokens[i]='-pro-'
            i+=1
        if val=='CD':
            tokens[i]='-num-'
    return tokens

def tokenize(lines):
    lines = lines.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    lines = word_tokenize(lines)
    lines = tag_pronoun(lines)
    # lines = [word.lower() for word in lines]
    return lines

def tokenize_sent(lines): # inpur is list of words
    lines = [word.strip("".join(punkt)) for word in lines]
    lines = [word for word in lines if len(word)>0]
    lines = tag_pronoun(lines)
    # lines = [word.lower() for word in lines]
    return lines

def tokenize_doc(lines):
    lines = nltk.sent_tokenize(lines)
    doc = []
    for line in lines:
        doc+=tokenize_sent(word_tokenize(line))
    return doc


if __name__ == '__main__':
	
	input_data_folder_path = sys.argv[1]
	model_path = sys.argv[2]
	pretrained_embeddings_path = sys.argv[3]

	print("Reading pre-trained embeddings . . ")
	word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=True)
	step0 = time()

	# curr_dir = os.getcwd()
	# dataset_dir = os.path.join(curr_dir, input_data_folder_path)
	print("Reading and pre-processing corpus . . .")
	corpus = get_corpus(input_data_folder_path)
	domain_model = get_domain_model(corpus, word2vec_model)
	
	print("Saving domain model in ", model_path, " . . . ")
	domain_model.wv.save(model_path)
	domain_model.save(model_path)





