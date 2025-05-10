# -*- coding: utf-8 -*-
"""
    This is to create the vocabularies which are used to convert the text data into tensor format
    Author: Thanh Vu <thanh.vu@csiro.au>
    Date created: 01/03/2019
    Date last modified: 19/03/2019
"""

import os
import torch
from collections import Counter
import numpy as np
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#===========================================================================
from src.util.mimiciii_data_processing import *
import pandas as pd


class Vocab(object):
    def __init__(self,
                 training_data: list,
                 training_labels: list,
                 min_word_frequency: int = -1,
                 max_vocab_size: int = -1,
                 word_embedding_mode: str = "word2vec",
                 word_embedding_file: str = None,
                 use_gpu: bool = True
                 ):
        """

        :param training_data:
        :param min_word_frequency:
        :param max_vocab_size:
        :param word_embedding_mode: str
            "word2vec": using word embeddings like word2vec or glove
            "fasttext": using subword embeddings like fastText
        :param word_embedding_file:
        :param use_gpu:
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.word_embedding_mode = word_embedding_mode
        self.word_embedding_file = word_embedding_file
        self.word_embedding_size = None
        self.word_embeddings = None

        self.training_data = training_data

        self.PAD_TOKEN = '_PAD'
        self.UNK_TOKEN = '_UNK'
        self.word2index = None
        self.index2word = None

        self.label2index = []
        self.index2label = []

        self.vocab_words = [self.PAD_TOKEN, self.UNK_TOKEN]
        self.all_labels = []

        self.min_word_frequency = min_word_frequency
        self.max_vocab_size = max_vocab_size

        self.logger = None
        self.update_labels(training_labels)
        self.init_title_emb()

    def prepare_vocab(self):
        self._build_vocab()

        # load pretrain word embeddings
        if self.word_embedding_file is not None:
            self.word_embeddings = torch.FloatTensor(self._load_embeddings())

    def index_of_word(self,
                      word: str) -> int:
        try:
            return self.word2index[word]
        except:
            return self.word2index[self.UNK_TOKEN]

    def index_of_label(self,
                       label: str, level: int) -> int:
        try:
            return self.label2index[level][label]
        except:
            return 0

    def _build_vocab(self):
        all_words = []

        for text, labels, _id in self.training_data:
            filtered_words = text.split()
            all_words.extend(filtered_words)

        counter = Counter(all_words)
        if self.max_vocab_size > 0:
            counter = {word: freq for word, freq in counter.most_common(self.max_vocab_size)}
        if self.min_word_frequency > 0:
            counter = {word: freq for word, freq in counter.items() if freq >= self.min_word_frequency}

        self.vocab_words += list(sorted(counter.keys()))
        #58008.150687
        print("vocab长度",len(self.vocab_words))
        #print(self.vocab_words)

        self.word2index = {word: idx for idx, word in enumerate(self.vocab_words)}
        self.index2word = {idx: word for idx, word in enumerate(self.vocab_words)}

    def update_labels(self, labels):
        self.all_labels = []
        self.index2label = []
        self.label2index = []
        for level_labels in labels:
            all_labels = list(sorted(level_labels))
            self.label2index.append({label: idx for idx, label in enumerate(all_labels)})
            self.index2label.append({idx: label for idx, label in enumerate(all_labels)})
            self.all_labels.append(all_labels)

    #=================================================================================

    def init_title_emb(self):
        self.tword2id,self.id2tword = {"<EOS>":0},["<EOS>"]

        mimicPath = "data/mimicdata/icd_title/"
        dIcdDiagnoses = pd.read_csv(os.path.join(mimicPath, 'D_ICD_DIAGNOSES.csv'), dtype=str)
        for dic, code in dIcdDiagnoses['ICD9_CODE'].items():
            # print(code,type(code),dIcdDiagnoses.loc[dic]['ICD9_CODE'])
            dIcdDiagnoses.loc[dic, 'ICD9_CODE'] = reformat(code, True, FULL)
        dIcdDiagnoses = dIcdDiagnoses.set_index('ICD9_CODE')
        dicdProcedures = pd.read_csv(os.path.join(mimicPath, 'D_ICD_PROCEDURES.csv'), dtype=str)
        dicdProcedures['ICD9_CODE'] = dicdProcedures['ICD9_CODE'].astype(str)
        for dic, code in dicdProcedures['ICD9_CODE'].items():
            dicdProcedures.loc[dic, 'ICD9_CODE'] = reformat(str(code), False, FULL)
        dicdProcedures = dicdProcedures.set_index('ICD9_CODE')
        icdTitles = pd.concat([dIcdDiagnoses, dicdProcedures])

        self.titles = []
        # id2icd列表记录了每一个icd
#         print("type(self.label2index)",type(self.label2index))
#         print("len(self.label2index)", len(self.label2index))
        # print("self.all_labels", self.all_labels)
#         print("self.label2index[0]", self.label2index[0])
        cnt = 1
        count = 0
        xx = 0
        for icd in self.label2index[0]:
            try:
                # 标签长描述和短描述拼接
                desc = (icdTitles.loc[icd]['SHORT_TITLE'] + ' <:> ' + icdTitles.loc[icd]['LONG_TITLE']).lower().split()
                xx += 1
            except:
                count += 1
                desc = " <:> ".split()
            self.titles.append(desc + ["<EOS>"])

            # print('desc[1]:',desc[1])

            # 从文本标题长和短描述中得到所有标题的所有词，
            for w in desc:
                if w not in self.tword2id:
                    self.tword2id[w] = cnt
                    self.id2tword.append(w)
                    cnt += 1
        print(count)
        print(xx)
        print(cnt)

        self.titleLen = [len(i) for i in self.titles]
        print("self.titleLen",self.titleLen)
        titleMaxLen = max(self.titleLen)
        print("titleMaxLen",titleMaxLen)
        self.twordNum = cnt
        self.tokenizedTitle = np.array(
            [[self.tword2id[w] for w in t] + [0] * (titleMaxLen - len(t)) for t in self.titles], dtype='int32')
#         self.totalSampleNum = len(self.tokenizedNote)

    #===========================================================================================================    
     
    def _load_embeddings(self):
        if self.word_embedding_file is None:
            return None

        gensim_format = False
        if self.word_embedding_file.endswith(".model") or self.word_embedding_file.endswith(".bin"):
            gensim_format = True

        if not gensim_format:
            if self.word_embedding_mode.lower() == "fasttext":
                return self._load_subword_embeddings()
            return self._load_word_embeddings()
        else:
            return self._load_gensim_format_embeddings()

    def _load_word_embeddings(self):
        embeddings = None
        embedding_size = None

        count = 0
        if not os.path.exists(self.word_embedding_file):
            raise Exception("{} is not found!".format(self.word_embedding_file))

        for line in open(self.word_embedding_file, "rt"):
            if count >= 0:

                split = line.rstrip().split(" ")
                word = split[0]
                vector = np.array([float(num) for num in split[1:]]).astype(np.float32)
                if len(vector) > 0:
                    if embedding_size is None:
                        embedding_size = len(vector)

                        unknown_vec = np.random.uniform(-0.25, 0.25, embedding_size)
                        embeddings = [unknown_vec] * (self.n_words())

                        embeddings[0] = np.zeros(embedding_size)
                    if word in self.word2index:
                        embeddings[self.word2index[word]] = vector
            count += 1

        self.word_embedding_size = len(embeddings[0])
        embeddings = np.array(embeddings, dtype=np.float32)

        return embeddings

    def _load_subword_embeddings(self):
        if not os.path.exists(self.word_embedding_file):
            raise Exception("{} is not found!".format(self.word_embedding_file))

        model = FastText.load_fasttext_format(self.word_embedding_file)
        embedding_size = model.wv["and"].size
        unknown_vec = np.random.uniform(-0.25, 0.25, embedding_size)

        embeddings = [unknown_vec] * (self.n_words())
        embeddings[0] = np.zeros(embedding_size)
        for word in self.word2index:
            try:
                embeddings[self.word2index[word]] = model.wv[word]
            except:
                # self.word2index[word] = self.word2index[self.UNK_TOKEN]
                pass

        self.word_embedding_size = len(embeddings[0])
        embeddings = np.array(embeddings, dtype=np.float32)

        return embeddings

    def _load_gensim_format_embeddings(self):
        if not os.path.exists(self.word_embedding_file):
            raise Exception("{} is not found!".format(self.word_embedding_file))

        if self.word_embedding_mode.lower() == "fasttext":

            if self.word_embedding_file.endswith(".model"):
                model = FastText.load(self.word_embedding_file)
            else:
                model = FastText.load_fasttext_format(self.word_embedding_file)

        elif self.word_embedding_file.endswith(".bin"):
            model = KeyedVectors.load_word2vec_format(self.word_embedding_file, binary=True)
        else:
            print("在这里")
            model = Word2Vec.load(self.word_embedding_file)

        print(model)
        embedding_size = model.wv["and"].size
        
        unknown_vec = np.random.uniform(-0.25, 0.25, embedding_size)

        embeddings = [unknown_vec] * (self.n_words())
        embeddings[0] = np.zeros(embedding_size)
        for word in self.word2index:
            try:
                embeddings[self.word2index[word]] = model.wv[word]
            except:
                # self.word2index[word] = self.word2index[self.UNK_TOKEN]
                pass

        self.word_embedding_size = len(embeddings[0])
        embeddings = np.array(embeddings, dtype=np.float32)

        return embeddings

    def n_words(self):
        return len(self.vocab_words)

    def n_labels(self, level):
        return len(self.all_labels[level])

    def n_level(self):
        return len(self.all_labels)

    def all_n_labels(self):
        output = []
        for level in range(len(self.all_labels)):
            output.append(len(self.all_labels[level]))
        return output
