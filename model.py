import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import gensim
from global_vars import *



class GensimModel(nn.Module):
    def __init__(self, model):
        super(GensimModel, self).__init__()
        self.weights = torch.FloatTensor(model.wv.vectors)

    def forward(self, word, model):
        word_embeds = model.wv[target].reshape(1,-1)
        return word_embeds 


class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, window_size, pretrained_weight=None):
        super(SimpleModel, self).__init__()
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.T1 = nn.Linear(in_features=2*window_size*embedding_dim, out_features=embedding_dim, bias=False)
        self.T2 = nn.Linear(in_features=embedding_dim, out_features=embedding_dim, bias=False)
        self.context_embedding.weight.requires_grad=False
        self.word_embedding.weight.requires_grad=False
        if pretrained_weight!=None:
            self.context_embedding.weight.data.copy_(torch.from_numpy(domain_model.wv.syn0))
            self.word_embedding.weight.data.copy_(torch.from_numpy(domain_model.syn1neg))
        
    def forward(self, context, words):
        context_embeds = self.context_embedding(context).view(1,-1) # 2cD*1
#         print("context embeds:", context_embeds.shape)
        h1 = self.T1(context_embeds) # 1*D
#         print("h1: ", h1.shape)
        h1 = torch.flatten(torch.t(h1)) #D
#         print("h1: ", h1.shape)
        h1 = h1.expand(words.shape[0],h1.shape[0]) #b*D
#         print("h1: ", h1.shape)
        word_embeds = self.word_embedding(words) # b*D
#         print("word embeds: ", word_embeds.shape)
        h2 = self.T2(word_embeds)
#         print("h1: ", h2.shape)
        score = torch.nn.CosineSimilarity(dim=1,eps=1e-8)(h1,h2) #b*1
#         print("score: ", score.shape)
        return score

