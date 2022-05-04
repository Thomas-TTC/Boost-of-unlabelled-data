import pickle
import pandas as pd
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression

def knn(k, learning):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    filename = 'sentence-transformers/train_emb.pkl'
    raw_data = open(filename, 'rb')
    data = pickle.load(raw_data)
    feature_set = data.loc[:,'TFIDF'].values.tolist()
    label_set = data.loc[:,'Sentiment'].values.tolist()

    knn_model.fit(feature_set, label_set)

    if learning == "semi-supervised":
        filename = 'sentence-transformers/unlabeled_emb.pkl'
        raw_data = open(filename, 'rb')
        data = pickle.load(raw_data)
        unlabeled_feature_set = data.loc[:, 'TFIDF'].values.tolist()
        unlabeled_label_set = [-1]*len(unlabeled_feature_set)

        feature_set = feature_set + unlabeled_feature_set
        label_set = label_set + unlabeled_label_set
        print(np.shape(feature_set))
        new_label_set = []

        for i in label_set:
            if i == 'positive':
                new_label_set.append(1)
            elif i == 'negative':
                new_label_set.append(0)
            else:
                new_label_set.append(-1)

        semi_knn_model = SelfTrainingClassifier(knn_model, criterion='k_best', k_best=10000, max_iter=10)
        semi_knn_model.fit(feature_set, new_label_set)

    filename = 'sentence-transformers/dev_emb.pkl'
    raw_data = open(filename, 'rb')
    data = pickle.load(raw_data)
    dev_feature_set = data.loc[:,'TFIDF'].values.tolist()
    dev_label_set = data.loc[:,'Sentiment'].values.tolist()

    if learning == 'semi-supervised':
        print(semi_knn_model.score(dev_feature_set, dev_label_set))
    else:
        print(knn_model.score(dev_feature_set, dev_label_set))

def lr():
    lr_model = LogisticRegression(penalty='none')

    

def main():
    args = sys.argv[1:]
    for arg in args:
        print(arg)
    knn(int(args[0]), args[1])

if __name__ == "__main__":
    main()