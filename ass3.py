import pickle
import pandas as pd
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

"""
train data with embedding data and given learning method, and return the evaluation results with development data set.
"""
def train(train_model, learning):
    labeled_feature_set, labeled_label_set = load_data('sentence-transformers/train_emb.pkl')

    if learning == 'supervised':
        train_model.fit(labeled_feature_set, labeled_label_set)

    elif learning == 'semi-supervised':
        unlabeled_feature_set, unlabeled_label_set = load_data('sentence-transformers/unlabeled_emb.pkl')
        feature_set = labeled_feature_set + unlabeled_feature_set
        label_set = labeled_label_set + unlabeled_label_set
        new_label_set = change_to_1_0(label_set)
        print(np.shape(new_label_set))

        semi_train_model = SelfTrainingClassifier(train_model, criterion='k_best', k_best=7000, max_iter=10)
        semi_train_model.fit(feature_set, new_label_set)

    dev_feature_set, dev_label_set = load_data('sentence-transformers/dev_emb.pkl')

    if learning == 'semi-supervised':
        predicted = change_back_from_1_0(semi_train_model.predict(dev_feature_set))
        confusion_matrix, accuracy, evaluations_df = evaluate(dev_label_set, predicted)
        print(confusion_matrix)
        print('accuracy: ' + str(accuracy) + '\n')
        print(evaluations_df)
        predict_test(semi_train_model, learning)
    else:
        predicted = train_model.predict(labeled_feature_set)
        confusion_matrix, accuracy, evaluations_df = evaluate(labeled_label_set, predicted)
        print(confusion_matrix)
        print('accuracy: ' + str(accuracy) + '\n')
        print(evaluations_df)

        predicted = train_model.predict(dev_feature_set)
        confusion_matrix, accuracy, evaluations_df = evaluate(dev_label_set, predicted)
        print(confusion_matrix)
        print('accuracy: ' + str(accuracy) + '\n')
        print(evaluations_df)
        predict_test(train_model, learning)


"""
Ensemble method that combines three weak models to create a strong model.
"""
def ensemble(model_1, model_2, model_3, learning):
    labeled_feature_set, labeled_label_set = load_data('sentence-transformers/train_emb.pkl')

    if learning == 'supervised':
        model_1.fit(labeled_feature_set, labeled_label_set)
        model_2.fit(labeled_feature_set, labeled_label_set)
        model_3.fit(labeled_feature_set, labeled_label_set)

    elif learning == 'semi-supervised':
        unlabeled_feature_set, unlabeled_label_set = load_data('sentence-transformers/unlabeled_emb.pkl')
        feature_set = labeled_feature_set + unlabeled_feature_set
        label_set = labeled_label_set + unlabeled_label_set
        new_label_set = change_to_1_0(label_set)
        print(np.shape(new_label_set))

        semi_1 = SelfTrainingClassifier(model_1, criterion='k_best', k_best=7000, max_iter=10)
        semi_1.fit(feature_set, new_label_set)
        semi_2 = SelfTrainingClassifier(model_2, criterion='k_best', k_best=7000, max_iter=10)
        semi_2.fit(feature_set, new_label_set)
        semi_3 = SelfTrainingClassifier(model_3, criterion='k_best', k_best=7000, max_iter=10)
        semi_3.fit(feature_set, new_label_set)

    dev_feature_set, dev_label_set = load_data('sentence-transformers/dev_emb.pkl')

    if learning == 'semi-supervised':
        predicted_1 = change_back_from_1_0(semi_1.predict(dev_feature_set))
        predicted_2 = change_back_from_1_0(semi_2.predict(dev_feature_set))
        predicted_3 = change_back_from_1_0(semi_3.predict(dev_feature_set))

        final_predict = []

        for i in range(0, len(predicted_1)):
            if predicted_1[i] == 'positive':
                if predicted_2[i] == 'positive' or predicted_3[i] == 'positive':
                    final_predict.append('positive')
                else:
                    final_predict.append('negative')
            else:
                if predicted_2[i] == 'negative' or predicted_3[i] == 'negative':
                    final_predict.append('negative')
                else:
                    final_predict.append('positive')
        confusion_matrix, accuracy, evaluations_df = evaluate(dev_label_set, final_predict)
        print(confusion_matrix)
        print('accuracy: ' + str(accuracy) + '\n')
        print(evaluations_df)
    else:
        predicted_1 = model_1.predict(dev_feature_set)
        predicted_2 = model_2.predict(dev_feature_set)
        predicted_3 = model_3.predict(dev_feature_set)

        final_predict = []

        for i in range(0, len(predicted_1)):
            if predicted_1[i] == 'positive':
                if predicted_2[i] == 'positive' or predicted_3[i] == 'positive':
                    final_predict.append('positive')
                else:
                    final_predict.append('negative')
            else:
                if predicted_2[i] == 'negative' or predicted_3[i] == 'negative':
                    final_predict.append('negative')
                else:
                    final_predict.append('positive')
        confusion_matrix, accuracy, evaluations_df = evaluate(dev_label_set, final_predict)
        print(confusion_matrix)
        print('accuracy: ' + str(accuracy) + '\n')
        print(evaluations_df)


"""
Return the predict result of trained model with test data set.
"""
def predict_test(model, learning):
    test_feature_set, test_label_set = load_data('sentence-transformers/test_emb.pkl')
    if learning == 'semi-supervised':
        predicted = change_back_from_1_0(model.predict(test_feature_set))
    else:
        predicted = model.predict(test_feature_set)

    identity = []
    for i in range(0, 4000):
        identity.append(i)
    data = {'Id': identity,
            'Category': predicted
            }

    df = pd.DataFrame(data, columns=['Id', 'Category'])

    df.to_csv(r'/Users/thomaschen/Documents/unimelb/Intro to ML/Assignments/A3/test.csv', index=False, header=True)

    print(df)


"""
Evaluation function to help the implementation.
"""
def evaluate(actual, predicted):
    # the confusion matrix
    labels_type = {'Actual': actual,
                   'Predicted': predicted
                   }

    df = pd.DataFrame(labels_type, columns=['Actual', 'Predicted'])

    confusion_matrix = pd.crosstab(df['Actual'], df['Predicted'], rownames=['Actual'], colnames=['Predicted'])

    accuracy = accuracy_score(actual, predicted)

    # evaluations of precision, recall and f_score
    actual = np.array(actual)
    predicted = np.array(predicted)
    evaluations = {'precision': [], 'recall': [], 'f_score': []}
    index = ['positive', 'negative']

    # among features
    precision, recall, f_score, support = precision_recall_fscore_support(actual, predicted, average=None, labels=index,
                                                                          zero_division=0)
    evaluations['precision'].extend(precision)
    evaluations['recall'].extend(recall)
    evaluations['f_score'].extend(f_score)

    # different types of average evaluations
    average_types = ["macro", "micro", "weighted"]

    for average_type in average_types:
        precision, recall, f_score, support = precision_recall_fscore_support(actual, predicted, average=average_type
                                                                              ,zero_division=0)
        index.append(average_type)
        evaluations['precision'].append(precision)
        evaluations['recall'].append(recall)
        evaluations['f_score'].append(f_score)

    evaluations_df = pd.DataFrame(evaluations, index=index)

    return confusion_matrix, accuracy, evaluations_df


"""
Load data from the given file name.
"""
def load_data(filename):
    raw_data = open(filename, 'rb')
    data = pickle.load(raw_data)
    feature_set = data.loc[:, 'TFIDF'].values.tolist()
    label_set = data.loc[:, 'Sentiment'].values.tolist()
    raw_data.close()

    return feature_set, label_set


"""
Change the label of positive and negative to 1 and 0.
"""
def change_to_1_0(label_set):
    new_label_set = []
    for i in label_set:
        if i == 'positive':
            new_label_set.append(1)
        elif i == 'negative':
            new_label_set.append(0)
        else:
            new_label_set.append(-1)

    return new_label_set


"""
Change the label back from 1 and 0 to positive and negative
"""
def change_back_from_1_0(label_set):
    new_label_set = []
    for i in range(0, len(label_set)):
        if label_set[i] == 1:
            new_label_set.append('positive')
        elif label_set[i] == 0:
            new_label_set.append('negative')

    return new_label_set


"""
Return the k-nearest neighbor model with given k parameter.
"""
def knn(k):
    knn_model = KNeighborsClassifier(n_neighbors=k, weights='distance')

    return knn_model


"""
Return the logistic regression model.
"""
def lr():
    lr_model = LogisticRegression(penalty='none',max_iter=10000)

    return lr_model


"""
Return the one hidden layer multi-layer perceptron with given number of neuron in hidden layer.
"""
def mlp(neurons):
    mlp_model = MLPClassifier(hidden_layer_sizes=(neurons, ), random_state=1, max_iter=1000, learning_rate_init=0.00005,
                              learning_rate='constant', activation='tanh')

    return mlp_model


"""
Execute the majority voting and return the result.
"""
def mv():
    labeled_feature_set, labeled_label_set = load_data('sentence-transformers/train_emb.pkl')
    positive = 0
    negative = 0
    for i in labeled_label_set:
        if i == 'positive':
            positive += 1
        else:
            negative += 1
    if positive > negative:
        major = 'positive'
    else:
        major = 'negative'

    dev_feature_set, dev_label_set = load_data('sentence-transformers/dev_emb.pkl')

    major_label_set = [major] * len(dev_label_set)
    confusion_matrix, accuracy, evaluations_df = evaluate(dev_label_set, major_label_set)
    print(confusion_matrix)
    print('accuracy: ' + str(accuracy) + '\n')
    print(evaluations_df)


def main():
    args = sys.argv[1:]
    # for i in range(1, 9):
    #     print(i)
    #     model = mlp(i)
    #     train(model, 'supervised')

    for arg in args:
        print(arg)
    if args[0] == 'knn':
        model = knn(int(args[2]))
    elif args[0] == 'lr':
        model = lr()
    elif args[0] == 'mlp':
        model = mlp(int(args[2]))
    elif args[0] == 'mv':
        mv()
        exit()
    elif args[0] == 'ensemble':
        model_1 = knn(5)
        model_2 = mlp(8)
        model_3 = lr()
        ensemble(model_1, model_2, model_3, args[1])
        exit()
    train(model, args[1])


if __name__ == "__main__":
    main()