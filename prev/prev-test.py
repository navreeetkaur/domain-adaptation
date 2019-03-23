import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm
import string
from collections import OrderedDict
import os
import math
import time
from random import randint
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.tokenize import word_tokenize

import gensim
from gensim.models import KeyedVectors
from util import *

punkt = set(list(string.punctuation)+["``","`"])

pad = False
lock_factor=0.9
window_size = 3
num_neg_samples = 10
num_epochs = 5
learning_rate = 0.001
min_count = 1

 def process_eval(df_row, vocab):
    sentence = df_row['sentence']
    sentence = sentence.split()
    idx = sentence.index('<<target>>')
    left_sentence = " ".join(sentence[:idx])
    right_sentence = " ".join(sentence[idx+1:])
    left_sentence = tokenize(left_sentence)
    left_sentence = replace_unk(left_sentence, vocab)
    right_sentence = tokenize(right_sentence)
    right_sentence = replace_unk(right_sentence, vocab)
    df_row['full_left'] = left_sentence
    df_row['full_right'] = right_sentence
    return df_row

def preprocess_targets(targets,vocab):
    td = []
    for i,target in enumerate(targets):
        target=target.lower()
        target = target.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        target = word_tokenize(target)   
        td.append(tag_pronoun(target))
        target = replace_unk(target, vocab)
        
    return td

# only for DEV
def target(df):
    target_true = df['target_word']
    idx = df['td'].index(target_true)
    df['idx'] = idx
    return df

def get_left_context(left_sentence):
    context_words_left = []
    for i in range(len(left_sentence)-1, len(left_sentence)-1-window_size,-1):
        try:
            context_words_left.append(left_sentence[i])
        except:
            if pad:
                context_words_left.append('LEFT_PAD')
    return context_words_left

def get_right_context(right_sentence):
    context_words_right = []
    for i in range(0,window_size):
        try:    
            context_words_right.append(right_sentence[i])
        except:
            if pad:
                context_words_right.append('RIGHT_PAD')
    return context_words_right

def mrr(ranks):
    print(len(ranks))
    mrr = 0
    for rank in ranks:
        mrr += 1./rank
    mrr = mrr / len(ranks)
    return mrr

def new_rank(df,domain_model):
    scores = []
    for target in df['td']:
        sim=0.0
        for token in target:
            t = domain_model.wv[token].reshape(1,-1)
            left_context = get_left_context(df['full_left'])
            for word in left_context:
                c = domain_model.wv[word].reshape(1,-1)
                sim+=cosine_similarity(t,c)[0][0]
            right_context = get_left_context(df['full_right'])
            for word in right_context:
                c = domain_model.wv[word].reshape(1,-1)
                sim+=cosine_similarity(t,c)[0][0]
        scores.append(sim/max(1,len(target)))
    ranks = np.argsort(np.array(scores)*-1)+1
    df['ranks'] = ranks
    # write_ranks(ranks, file)
    # df['eval_rank'] = ranks[df['idx']] # DEV only
    # print(ranks[df['idx']]) # DEV only
    return df

def write_output(df, file):
    with open(file, 'w') as f:
        for index, row in df.iterrows():
            to_write = ' '.join(map(str, row['ranks']))
            to_write=to_write.replace('[',"")
            to_write=to_write.replace(']',"")
            to_write=to_write.replace(',',"")
            f.write(to_write+'\n')

def write_ranks(ranks, file):
    with open(file, 'a') as f:
        to_write = ' '.join(map(str, ranks))
        to_write=to_write.replace('[',"")
        to_write=to_write.replace(']',"")
        to_write=to_write.replace(',',"")
        f.write(to_write+'\n')

if __name__ == '__main__':
    evaluation_txt_file_path = sys.argv[1]
    evaluation_txt_td_file_path = sys.argv[2]
    model_path = sys.argv[3]
    outfile = 'output.txt'

    print("loading model . . ")
    domain_model = KeyedVectors.load(model_path)
    curr_vocab = set(domain_model.wv.vocab.keys())

    sentences = []
    target_words = []
    target_dict = []
    print("reading eval data - gold")
    with open(evaluation_txt_file_path, 'r') as f:
        for line in f:
            line = line.strip().split("::::")
            sentences.append(line[0])
            target_words.append(line[1])
    print("reading eval data - td")
    with open(evaluation_txt_td_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            target_dict.append(line)

    df = pd.DataFrame({'sentence':sentences,'td':target_dict})
    print("Processing data for ranking")
    # df = df.progress_apply(lambda x: target(x),axis=1) # for dev only

    df['td'] = df['td'].progress_apply(lambda x: preprocess_targets(x,curr_vocab))

    df = df.progress_apply(lambda x: process_eval(x, curr_vocab), axis=1) 
    df = df.drop(columns=['sentence'])
    print("Calculating ranks")
    df = df.apply(lambda x: new_rank(x,domain_model), axis=1)
    print("Writing output")
    write_output(df,outfile)


