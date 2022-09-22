# -*- coding:utf-8 -*-

"""



Author:

    Weichen Shen,weichenswc@163.com



Reference:

    [1] Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations[C]//Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014: 701-710.(http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)



"""
import networkx as nx
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

def get_randomwalk(self,node,path_length):
    
     random_walk = [node]
     for i in range(path_length-1):
            temp = list(self.graph.neighbors(node))
            temp = list(set(temp) - set(random_walk))    
            if len(temp) == 0:
                break

     random_node = random.choice(temp)
     random_walk.append(random_node)
     node = random_node
     return random_walk
def fit_walks(self):
    random_walks=[]
    all_nodes = list(self.graph.nodes())
    for n in tqdm(all_nodes):
            for i in range(5):
                random_walks.append(get_randomwalk(self,n,10))

    return random_walks
class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}
        self.random_walks=[]


    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = str(self.sentences)
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter
        print("Fit random walks...")
        self.random_walks=fit_walks(self)
        print("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        model.build_vocab(self.random_walks, progress_per=2)
        self.w2v_model = model.train(self.random_walks, total_examples = model.corpus_count, epochs=20, report_delay=1)
        print("Learning embedding vectors done!")
        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings
