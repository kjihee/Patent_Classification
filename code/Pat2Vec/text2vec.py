import re
import random
import glove
import csv
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, LabeledSentence

class text2vec(object):
    def __init__(self, filename, model, class_name, embed_dir=None, previous_model=None):
        self._filename = filename
        # self._labels = labels
        self.class_name = class_name
        self._previous_model = previous_model
        self._y_idx = None
        self._sentences = None
        self._labels = None
        self._embed_dir = embed_dir

    def text_preprocessing(self):
        df = pd.read_csv(self._filename)
        abstract = df["abstract"]
        self._labels = df['label']

        abstract = [re.sub('[=.,#/?:$}]', '', text.lower()) for text in abstract]

        stop_words = set(stopwords.words('english'))
        word_tokens = [word_tokenize(text) for text in abstract]

        filtered_sentence = [[w for w  in row if not w in stop_words] for row in word_tokens]
        cleaned_text = [' '.join(row) for row in filtered_sentence]

        return cleaned_text

    def vectorize(self, text):
        vectorizer = CountVectorizer(min_df=2)
        vectorizer.fit(text)

        X = vectorizer.fit_transform(text)
        Xc = X * X.T
        Xc.setdiag(0)

        result = Xc.toarray()

        dic = {idx1 : {idx2: word2 for idx2, word2 in enumerate(word1) if word2 > 0}
                 for idx1, word1 in enumerate(result)}

        return dic

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
        dic = self.tfidf_vectorize(cleaned_text)

        model = glove.Glove(dic, d=300, alpha=0.75, x_max=100.0)

        for epoch in range(150):
            err = model.train(batch_size=150, workers=3)
            # print("epoch %d, error %.3f" % (epoch, err), flush=True)

        X = model.W
        #self.save_embed_file('embed_{}/{}.embd'.format(self._embed_dir, self.class_name), X, self._labels)

        return model.W

    def train_doc2vec(self):
        cleaned_text = self.text_preprocessing()

        self._sentences = [LabeledSentence(cleaned_text[idx], [str(self._labels[idx])])
                        for idx in range(len(cleaned_text))]

        model = Doc2Vec(alpha=0.025, window=9, min_count=1, min_alpha=0.025, size=300)
        model.build_vocab(self._sentences)

        if self._previous_model is not None:
            model.intersect_word2vec_format(
                'embed_{}/{}.embd'.format(self._embed_dir, self._previous_model), lockf=1.0, binary=True
            )

        for i in range(25):
            random.shuffle(self._sentences)
            model.alpha -= 0.001
            model.min_alpha = model.alpha
            model.train(self._sentences, total_examples=model.corpus_count, epochs=model.iter)

        # model.save('model/abstract.doc2vec')

        X = [model.docvecs[i] for i in self._labels]
        self.save_embed_file('embed_{}/{}.embd'.format(self._embed_dir, self.class_name), X, self._labels)
        return X

    def save_embed_file(self, save_path, X, labels):
        with open(save_path,'w') as f:
            f.write("%s %s\n"%(len(X), 300))
            for i in labels:
                f.write(str(i) +" "+" ".join([str(x) for x in X[i]]))
