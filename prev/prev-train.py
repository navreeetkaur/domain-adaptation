import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm
import string
from collections import OrderedDict
import os
import math
from time import time
from random import randint

import nltk
from nltk.tokenize import word_tokenize

import gensim
from util import *

pad = False
lock_factor=0.9
window_size = 3
num_neg_samples = 10
num_epochs = 5
learning_rate = 0.001
min_count=1
punkt = set(list(string.punctuation)+["``","`"])

def get_corpus(dataset_dir):
	corpus = []
	for filename in os.listdir(dataset_dir):
	    print(filename)
	    with open(os.path.join(dataset_dir,filename), 'r') as f:
	        lines = f.readlines()
	        lines = " ".join(lines)
	        lines = tokenize_doc(lines)
	        # print(len(lines))
	        corpus+=(lines)
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
	embedding_dim = word2vec_model.wv.vectors[0].shape[0]
	domain_model = gensim.models.Word2Vec(size=embedding_dim, alpha=0.025,
                   window=5, min_count=1, max_vocab_size=None, 
                   sample=0.001, workers=4, min_alpha=0.0001, 
                   sg=1, hs=0, negative=25, cbow_mean=1, iter=5, 
                   null_word=0, trim_rule=None, sorted_vocab=1, batch_words=1000)
	domain_model.build_vocab(corpus)
	total_examples = domain_model.corpus_count
	domain_model.build_vocab([list(word2vec_model.vocab.keys())], update=True)
	domain_model.intersect_word2vec_format(pretrained_embed_path, binary=True, lockf=lock_factor)
	domain_model.train(corpus, total_examples=total_examples, epochs=domain_model.epochs)
	return domain_model

if __name__ == '__main__':
	input_data_folder_path = sys.argv[1]
	model_path = sys.argv[2]
	pretrained_embeddings_path = sys.argv[3]

	print("Reading pre-trained embeddings . . ")
	word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=True)

	step0 = time()

	curr_dir = os.getcwd()
	dataset_dir = os.path.join(curr_dir, input_data_folder_path)
	
	# read corpus
	print("Reading and pre-processing corpus . . .")
	corpus = get_corpus(dataset_dir)
	
	step1 = time() 
	print('Time: {:5.2f} '.format(step1-step0))

	# get domain-specific vocab
	print("Building domain-vocab . . .")
	counter = get_counter(corpus)
	word2idx = build_vocab(counter, threshold=min_count)
	idx2word = OrderedDict([(v, k) for k, v in word2idx.items()])
	step2 = time()
	print('Time: {:5.2f} '.format(step2-step1))

	# get combined vocab
	print("Combining with word2vec vocab . . .")
	combined_vocab = combine_vocab(model=word2vec_model,domain_vocab=word2idx)
	combined_vocab = set(combined_vocab)
	step3 = time()
	print('Time: {:5.2f} '.format(step3-step2))

	# print("Replacing OOV with UNK . . .")
	# for i,doc in enumerate(corpus):
	# 	# print(i)
	# 	for j,word in enumerate(doc):
	# 		if word not in combined_vocab:
	# 			corpus[i][j]='UNK'

	step4 = time()
	print('Time: {:5.2f} '.format(step4-step3))

	# # get word2idx, idx2word for combined vocab
	# print("Building look-up for bigger vocab . . .")
	# word2idx_big = OrderedDict(zip(list(combined_vocab), [i for i in range(len(combined_vocab))]))
	# idx2word_big = OrderedDict([(v, k) for k, v in word2idx_big.items()])

	print("Building domain model . . . ")
	domain_model = get_domain_model(corpus, word2vec_model)

	print("Saving domain model . . . ")
	domain_model.wv.save(model_path)

	# if MODE==0:
	# 	print("Saving torch model . . . ")
	# 	gensim_model = GensimModel(domain_model)
	# 	torch.save(gensim_model, model_path)


	# if MODE==1:
	# 	print("Building unigram-table for negative sampling . . .")
	# 	word2freq = make_word2freq(counter, word2idx)
	# 	simplemodel = SimpleModel(vocab_size=len(combined_vocab), embedding_dim=embedding_dim, window_size=window_size)
	# 	optimizer = optim.Adam(simplemodel.parameters(), lr=learning_rate)
	# 	print("Trainable parameters: ")
	# 	for name, param in simplemodel.named_parameters():
	# 	    if param.requires_grad:
	# 	        print(name)
	# 	simplemodel.zero_grad()

	# 	## - - - - TRAINING - - - - -

	# 	print("Starting training. . .")
	# 	simplemodel.train()
	# 	total_loss = 0.
	# 	start_time = time.time()

	# 	for epoch in range(1,num_epochs+1):
	# 	    print("- - - - - - - - - - - - - - - - - - - -")
	# 	    for j,doc in enumerate(corpus):
	# 	        print("~~~~~~~")
	# 	        data = form_pairs(doc,word2idx=word2idx_big, window_size=3)
	# 	        curr_loss=0.0
	# 	        for i, (context, target) in enumerate(data):
	# 	            centre_words = [[target,1]] + [[target,-1] for target in negative_sample(table, num=num_neg_samples, idx2word=idx2word, word2idx_big=word2idx_big)]
	# 	            centre_words = torch.LongTensor(centre_words)
	# 	            words = torch.flatten(centre_words[:,0])
	# 	            targets = torch.flatten(centre_words[:,1]).type('torch.FloatTensor')
	# 	            context = torch.LongTensor(context)
	# 	            score = simplemodel(context=context, words=words)
	# 	            #loss = loss_fn(score=score, target=targets)
	# 	            s_true, s_neg = generate_x1_x2(score)
	# 	            loss = nn.MarginRankingLoss(reduction='mean', margin=1.0)(input1=s_true, input2=s_neg, target=targets)
	# 	            optimizer.zero_grad()
	# 	            loss.backward()

	# 	            total_loss += loss.item()
	# 	            optimizer.step()
	# 	            curr_loss+=loss.item()

	# 	            if i%10000==0:
	# 	                elapsed = time.time() - start_time
	# 	                print('- | epoch {:3d} | {:5d}/{:5d} words | lr {:02.3f} | '
	# 	                        'loss {:5.2f}  - '.format(
	# 	                    epoch, i, len(doc), learning_rate, curr_loss))
	# 	                start_time = time.time()
	# 	        print("\nLoss :", curr_loss/i, "  ; Time: ",elapsed,"\n")

	# 	torch.save(simplemodel, model_path)







