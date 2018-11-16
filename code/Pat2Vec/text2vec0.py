import re
import random
import glove
import pickle
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, LabeledSentence

class text2vec(object):
    def __init__(self, filename, model, class_name, previous_model=None, embed_dir='embed', n_dim=300):
        self._filename = filename
        # self._labels = labels
        self.class_name = class_name
        self._model = model
        self._previous_model = previous_model
        self._y_idx = None
        self._sentences = None
        self._labels = None
        self._embed_dir = embed_dir
        self._n_dim = n_dim

    def text_preprocessing(self):
        df = pd.read_pickle(self._filename)
        abstract = df["text"]
        self._labels = df['label']
        
        re_punc = re.compile("[%s]" % re.escape(string.punctuation))
        text_concat = [re_punc.sub("", text.lower()) for text in abstract]

        stop_words = set(stopwords.words('english'))
        word_tokens = [word_tokenize(text) for text in text_concat]

        filtered_sentence = [[w for w  in row if (w not in stop_words) or (w.strip() != "")] for row in word_tokens]
        
        if self._model == 'glove':
            cleaned_text = [' '.join(row) for row in filtered_sentence]
        else:
            cleaned_text = [row for row in filtered_sentence]

        return cleaned_text

    def count_vectorize(self, text):
        vectorizer = CountVectorizer(min_df=2)
        vectorizer.fit(text)

        X = vectorizer.fit_transform(text)
        Xc = X * X.T
        Xc.setdiag(0)

        result = Xc.toarray()

        dic = {idx1 : {idx2: word2 for idx2, word2 in enumerate(word1) if word2 > 0}
                 for idx1, word1 in enumerate(result)}

        return dic

    def tfidf_vectorize(self, text):
        vectorizer = TfidfVectorizer(min_df=2)
        vectorizer.fit(text)

        X = vectorizer.fit_transform(text)
        Xc = X * X.T
        Xc.setdiag(0)

        result = Xc.toarray()

        dic = {idx1 : {idx2: word2 for idx2, word2 in enumerate(word1) if word2 > 0}
                 for idx1, word1 in enumerate(result)}

        return dic


    def train_glove(self):

        cleaned_text = self.text_preprocessing()
        dic = self.count_vectorize(cleaned_text)

        model = glove.Glove(dic, d=self._n_dim, alpha=0.75, x_max=100.0)

        for epoch in range(150):
            err = model.train(batch_size=150, workers=3)
            # print("epoch %d, error %.3f" % (epoch, err), flush=True)

        X = model.W
       # self.save_embed_file('embed_{}/{}.embd'.format(self._embed_dir, self.class_name), X, self._labels)

        return model, X

    def train_doc2vec(self):
        cleaned_text = self.text_preprocessing()
        
        self._sentences = [LabeledSentence(cleaned_text[idx], [str(self._labels[idx])])
                        for idx in range(len(cleaned_text))]

        model = Doc2Vec(alpha=0.025, window=9, min_count=1, min_alpha=0.025, size=self._n_dim)
        model.build_vocab(self._sentences)

        if self._previous_model is not None:
            model.intersect_word2vec_format(
                'embed_{}/{}.embd'.format(self._embed_dir, self._previous_model), lockf=1.0, binary=True
            )

        for i in range(50):
            random.shuffle(self._sentences)
            model.alpha -= 0.0005
            model.min_alpha = model.alpha
            model.train(self._sentences, total_examples=model.corpus_count, epochs=model.iter)

        model.save('model/obesity_text_ver.doc2vec')

        X = [model.docvecs[str(i)] for i in self._labels]
        #self.save_embed_file('embed_{}/{}.embd'.format(self._embed_dir, self.class_name), X, self._labels)
        
        return model, X

    def save_embed_file(self, save_path, X, labels):
        with open(save_path,'w') as f:
            f.write("%s %s\n"%(len(X), self._n_dim))
            for i in labels:
                f.write(str(i) +" "+" ".join([str(x) for x in X[i]]))
