import random
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from gensim.models.word2vec import Word2Vec
from collections import defaultdict
import glove


class network2vec(object):
    def __init__(self, labels, filename, class_name, model,
                        previous_model=None, is_network=True, embed_dir='embed', n_dim=300):
        self._labels = labels
        self._model = model
        self._filename = filename
        self.class_name = class_name
        self._previous_model = previous_model
        self._is_network = is_network
        self._y_idx = None
        self._embed_dir = embed_dir
        self._n_dim = n_dim

    def preprocessing(self, filename):

        df = pd.read_csv(filename, sep=',', header = None, names=["idx","cited"])
        df["values"] = 1

        if self._is_network is True and self._model != 'glove':
            df = df.groupby(["idx", "cited"])["values"].sum().unstack().fillna(0)
            citation = np.array(df.values, int)
            r_citation = [[str(idx2) for idx2, word2 in enumerate(word1) if word2 > 0]
                for idx1, word1 in enumerate(citation)]

        elif self._is_network is True and self._model == 'glove':
            df = df.groupby(["idx", "cited"])["values"].sum().unstack().fillna(0)
            citation = np.array(df.values, int)

        elif self._is_network is False and self._model != 'glove':
            df['cited'] = df['cited'].astype(np.str)
            df = df.groupby(['idx'])['cited'].apply(lambda x: list(x))
            r_citation = df.tolist()

        elif self._is_network is False and self._model == 'glove':
            df = df.groupby(["idx", "cited"])["values"].sum().unstack().fillna(0).astype(int)
            X = df.values
            Xc = np.dot(X,X.T)
            Xc[np.diag_indices_from(Xc)] = 0
            citation = Xc

        df_idx = list(df.index.unique().values)
        missing_idx = list(set(self._labels.values) - set(df_idx))

        if self._model == 'glove':
            processed_citation = {idx1: {idx2:word2 for idx2, word2 in enumerate(word1) if word2 > 0}
                    for idx1, word1 in zip(df_idx, citation)}

        elif self._model == 'doc2vec':
            processed_citation = [LabeledSentence(" ".join(r_citation[idx]),[str(df_idx[idx])])
                    for idx in range(len(df_idx))]


            processed_citation = sorted(processed_citation, key=lambda ipc: int(ipc.tags[0]))

        elif self._model == 'word2vec':
            processed_citation = [list(set([str(idx)] + cited))
                    for i, (idx, cited) in enumerate(zip(df_idx, r_citation))]

        return processed_citation


    def train_word2vec(self):
        neighbors = self.preprocessing(self._filename)

        p2v = Word2Vec(alpha=0.025, window=8, min_count=1, min_alpha=0.020, size=self._n_dim)
        p2v.build_vocab(neighbors)
        print(neighbors)
        if self._previous_model is not None:
            p2v.intersect_word2vec_format(
                    'embed_{}/{}.embd'.format(self._embed_dir, self._previous_model), lockf=1.0, binary=True
            )

        for i in range(50):
            random.shuffle(neighbors)
            p2v.alpha -= 0.0005
            p2v.min_alpha = p2v.alpha
            p2v.train(neighbors, total_examples=p2v.corpus_count, epochs=p2v.iter)
        
        X = [p2v[str(i)] for i in range(len(neighbors))]
        return p2v, X


    def train_doc2vec(self):
        processed_sentence = self.preprocessing(self._filename)

        p2v = Doc2Vec(alpha=0.025, window=9, min_count=1, min_alpha=0.020, size=self._n_dim)
        p2v.build_vocab(processed_sentence)
        print(processed_sentence)
        if self._previous_model is not None:
            p2v.intersect_word2vec_format(
                    'embed_{}/{}.embd'.format(self._embed_dir, self._previous_model), lockf=1.0, binary=True
            )

        for i in range(50):
            random.shuffle(processed_sentence)
            p2v.alpha -= 0.0005
            p2v.min_alpha = p2v.alpha
            p2v.train(processed_sentence,total_examples=p2v.corpus_count,epochs=p2v.iter)

        X = [p2v.docvecs[i] for i in range(len(processed_sentence))]
        return p2v, X

    def train_glove(self):
        processed_sentence = self.preprocessing(self._filename)
        model = glove.Glove(processed_sentence, d=self._n_dim, alpha=0.75, x_max=100.0)

        for epoch in range(150):
            err = model.train(batch_size=150, workers=3)

        X = model.W
        return model, X

    def get_labels(self):
        return self._labels
