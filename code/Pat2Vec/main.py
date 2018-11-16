from text2vec import text2vec
from network2vec import network2vec
import numpy as np
import pandas as pd
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.cluster import KMeans
from datetime import datetime
import sys
import os
import shutil
from optparse import OptionParser

MAX_NUMBER_OF_PARSE = 10
process_id = 'embed'

def predict(X):
    # kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    # y_predict = kmeans.predict(X)

    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
            n_components=3, covariance_type='diag').fit(X)

    y_predict = gmm.predict(X)

    return y_predict

def accuracy(y_predict, Y):
    y_true=np.array(Y)
    assert y_predict.size == y_true.size

    D = max(y_predict.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_predict.size):
        w[y_predict[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    accuracy = sum([w[i, j] for i, j in ind])*1.0/y_predict.size, w

    return accuracy[0]


def select_model(data, labels, previous=None, embed_dir='embed'):

    X = None
    if data[0] == 'abstract':

        model = text2vec(filename=data[1], class_name=data[0], model=data[2], previous_model=previous, embed_dir=embed_dir) #, embed_dir=
        X = eval("model.train_{}()".format(data[2]))
    else:
        print(data[0])
        file_name = data[1].split('/')[-1]
        print("file: ", file_name)
        is_network = False if 'no' in file_name else True

        model = network2vec(labels=labels, filename=data[1], class_name=data[0], model=data[2],
                    previous_model=previous, is_network=is_network, embed_dir=embed_dir)
        X = eval("model.train_{}()".format(data[2]))

    return X

def main(argv):

    df1 = pd.read_csv('./ship_text.tsv', sep = '\t')
    df1["label"] = '1'

    df2 = pd.read_csv('./obesity_text.tsv', sep = '\t')
    df2["label"] = '2'

    df3 = pd.read_csv('./radio_text.tsv', sep = '\t')
    df3["label"] = '3'
    frames = [df1, df2, df3]
    df = pd.concat(frames)
    # df.to_csv("abstract_all.tsv")
    abstract = df["abstract"]
    labels = df['patent_id']
    Y = df["label"]

    parser = OptionParser()

    parser.add_option('--proc', dest='proc')

    for i in range(MAX_NUMBER_OF_PARSE):
        parser.add_option("--{}".format(i), dest="{}".format(i), nargs=3,\
            help="""Please Input data_type, file_path and model \n
                ex) abstract data/abstract.pkl doc2vec""")

    (options, args) = parser.parse_args()
    option_value = vars(options)
    proc_id = option_value['proc']
    embed_path = './embed_{}'.format(proc_id)

    if os.path.exists(embed_path):
        shutil.rmtree(embed_path)
    os.mkdir(embed_path)

    option_list = []

    for i in range(MAX_NUMBER_OF_PARSE):
        if option_value[str(i)] != None:
            option_list.append(option_value[str(i)])

    model_name = [i[0] for i in option_list]

    Xs = None

    try:
        outputs = []
        # option_list=[[abstract], './abstract_all.tsv' ,['./ship_text.tsv']]
        for idx, data in enumerate(option_list):
            if idx == 0:

                X = select_model(data, labels=labels, embed_dir=proc_id)
                Xs = np.array(X)
                print("1", Xs)
                y_predict = predict(X)

                data = list(data)
                data[1] = data[1].split('/')[-1]
                outputs.append(" ".join(data)+ "," +str(round(accuracy(y_predict, Y)*100, 2)))
                # print(" ".join(data) + "," +str(round(accuracy(y_predict, Y)*100, 2))
            else:
                X = select_model(data, labels=labels, previous=model_name[idx-1], embed_dir=proc_id)
                if Xs is None:

                    Xs = np.array(X)
                    print("2", Xs)
                else:
                    Xs = np.concatenate((Xs, X), axis=1)
                    print("3", Xs)
                y_predict = predict(X)

                data = list(data)
                data[1] = data[1].split('/')[-1]
                outputs.append(" ".join(data)+ "," +str(round(accuracy(y_predict, Y)*100, 2)))
            # print(" ".join(data) + "," +str(round(accuracy(y_predict, Y)), 2))
        y_predict = predict(Xs)
        print(str(",".join(outputs))+","+str(round(accuracy(y_predict, Y)*100, 2)))
    # print("total score :", accuracy(y_predict, Y))
    # except UnboundLocalError:
    except UnboundLocalError:
        print(str(",".join(outputs))+", UnboundLocalError")
    finally:
        shutil.rmtree('./embed_{}'.format(proc_id))

if __name__ == "__main__":
    #os.mkdir('./embed')
    #os.mkdir('./model')
    main(sys.argv[1:])
    #
    #shutil.rmtree('./embed_{}'.format(process_id))
    # shutil.rmtree('./model')
