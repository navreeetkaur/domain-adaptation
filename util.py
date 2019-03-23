import nltk
from nltk.tokenize import word_tokenize 
import gensim
import string
from collections import OrderedDict
import string

punkt = set(list(string.punctuation)+["``","`"])

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
    lines = [word.strip("".join(punkt)) for word in lines if word not in punkt]
    lines = tag_pronoun(lines)
    # lines = [word.lower() for word in lines]
    return lines

def tokenize_doc(lines):
    lines = nltk.sent_tokenize(lines)
    lines = [tokenize_sent(word_tokenize(line)) for line in lines]
    return lines


def get_counter(corpus):
    counter = {}
    for i,doc in enumerate(corpus):
        for j,word in enumerate(doc):
            counter[word] = counter.get(word,0) + 1
        print(j," words in doc ", i)
    return sorted(counter.items(), key=lambda pair:pair[1], reverse=True)


def build_vocab(counter, threshold):
    vocab = OrderedDict()
    vocab['UNK']=0
    for word, freq in counter:
        if freq<threshold:
            print(len(vocab))
            return vocab
        vocab[word] = len(vocab)
    print(len(vocab))
    return vocab


def replace_unk(words, vocab):
    for i,word in enumerate(words):
        if word not in vocab:
            words[i]='UNK'
    return words


def mrr(ranks):
    mrr = 0
    for rank in ranks:
        mrr+=float(1/rank)
    mrr = mrr/len(ranks)
    return mrr

    