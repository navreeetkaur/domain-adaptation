import numpy as np
import pandas as pd
import string
import os
import sys
import math
import time
from random import randint
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize 
import gensim
from gensim.models import KeyedVectors
from scipy.stats import rankdata

punkt = list(string.punctuation)+["``","`"]

def process_eval(df_row):
    sentence = df_row['sentence']
    sentence = sentence.split()
    idx = sentence.index('<<target>>')
    left_sentence = " ".join(sentence[:idx])
    right_sentence = " ".join(sentence[idx+1:])
    left_sentence = tokenize_sent(word_tokenize(left_sentence))
    right_sentence = tokenize_sent(word_tokenize(right_sentence))
    df_row['full_left'] = left_sentence
    df_row['full_right'] = right_sentence
    return df_row


def tag_pronoun(tokens):
    tags = nltk.pos_tag(tokens)
    for i,(key,val) in enumerate(tags):
        if val=='NNP' or val=='NNPS':
            tokens[i]='-pro-'
            i+=1
        if val=='CD':
            tokens[i]='-num-'
    return tokens


def tokenize_sent(lines): # inpur is list of words
    lines = [word.strip("".join(punkt)) for word in lines]
    lines = [word for word in lines if len(word)>0]
#     lines = tag_pronoun(lines)
#     lines = [word.lower() for word in lines]
    return lines


def preprocess_targets(targets,vocab):
    for i,target in enumerate(targets):
#         target = target.lower()
        target = target.strip("".join(punkt))
        targets[i]=target
    return targets

def rank(row, model, curr_vocab):
#     idx = row['idx']
    context =  row['full_left']+row['full_right']
    context_embeds = np.zeros(300).reshape(1,-1)
    for c in context:
        if c in curr_vocab:
            context_embeds += model.wv[c].reshape(1,-1)
    
    target_embeds = [model.wv[target].reshape(1,-1) if target in curr_vocab else np.zeros(300).reshape(1,-1) for target in row['td']]
    sim = [cosine_similarity(t,context_embeds)[0][0] for t in target_embeds]
    
    ranks = rankdata(np.array(sim)*-1, method='ordinal')
    if(len(row['td'])!=len(ranks)):
        print("ERRORRR")
#     row['eval_rank'] = ranks[idx]
    row['ranks'] = ranks
#     write_ranks(ranks,'outputfile.txt')
    return row

def write_output(df, file):
    with open(file, 'w') as f:
        for index, row in df.iterrows():
            to_write = ' '.join(map(str, row['ranks']))
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
    model = KeyedVectors.load(model_path)
    curr_vocab = set(model.wv.vocab.keys())
    print("Vocab size: ", len(curr_vocab))

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
    print("Preprocessing sentences . . ")
    df = df.apply(lambda x: process_eval(x),axis=1)
    df = df.drop(columns=['sentence'])
    print("Preprocessing target dictionary . . . ")
    df['td'] = df['td'].apply(lambda x: preprocess_targets(x,curr_vocab))
    print("Calculating ranks . . .")
    df = df.apply(lambda x: rank(x,model,curr_vocab),axis=1)
    print("Writing output")
    write_output(df,outfile)





