"""
Shows training accuracy of decision tree, support vector classifier
and neural network classifier via scikit - learn.
"""
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import pandas as pd

__author__ = "Alicem Murat ÖCAL"


def show_accuracy_of_methods(method_name="dt"):
    """
    Show accuracy of methods. for decision tree and support vector classifier
    and neural network.

    :param method_name: name of methods.
    'dt' for decision tree
    'svc' for support vector classifier.
    'nn' for neural network
    """

    # read data
    data = pd.read_csv("data/train.csv", sep=",")

    # split dataset as x and y
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]

    # normalize features by using mean normalization
    for i in X:
        X[i] = (X[i] - X[i].mean())/X[i].std()

    # convert pandas data frame to numpy.ndarray
    X = X.values

    # decision tree learner
    if method_name == "dt":

        # feature selection, get 200 best features applying by F-Test score.
        X = SelectKBest(score_func=f_classif, k=200).fit_transform(X, y)

        # select classifier as decision tree classifier of scikit-learn library.
        classifier = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=6,
                                            max_features='sqrt', presort=False)
    # support vector classifier
    elif method_name == "svc":

        # feature selection, get 1100 best features applying by F-Test score.
        X = SelectKBest(score_func=f_classif, k=1100).fit_transform(X, y)

        # select classifier as support vector classifier of scikit-learn library.
        classifier = SVC(C=10, kernel='rbf', degree=2, gamma=0.0001)

    # neural network
    elif method_name == "nn":

        # feature selection, get 200 best features applying by F-Test score.
        X = SelectKBest(score_func=f_classif, k=200).fit_transform(X, y)

        # select classifier as neural network classifier of scikit-learn library.
        classifier = MLPClassifier(activation='relu', max_iter=900, solver='adam', learning_rate='constant',
                                   momentum=0.9, batch_size=10, alpha=0.1)

    # split data as training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # convert pandas data frame to numpy.ndarray
    y_train = y_train.values
    y_test = y_test.values

    # train classifier with training data
    classifier = classifier.fit(X_train, y_train)

    # predict results of test data
    # list of predictions that contains the results of test data.
    predictions = classifier.predict(X_test)

    # show results for selected learning methods
    if method_name == "dt":

        # summarize performance of the classification
        print('\nThe overall accuracy score of decision tree classifier is: ' +
              str(accuracy_score(y_test, predictions)))

        print('\nThe overall f1 score of decision tree classifier is: ' + str(f1_score(y_test, predictions)))

        print('\nThe overall roc_auc score of decision tree classifier is: ' + str(roc_auc_score(y_test, predictions)))

        report = classification_report(y_test, predictions, target_names=['0', '1'])
        print('\n\nA detailed classification report for decision tree classifier: \n\n' + report)

    elif method_name == "svc":

        # summarize performance of the classification
        print('\nThe overall accuracy score of support vector classifier is: ' +
              str(accuracy_score(y_test, predictions)))

        print('\nThe overall f1 score of support vector classifier is: ' + str(f1_score(y_test, predictions)))

        print('\nThe overall roc_auc score of support vector classifier is: ' + str(roc_auc_score(y_test, predictions)))

        report = classification_report(y_test, predictions, target_names=['0', '1'])
        print('\n\nA detailed classification report for support vector classifier: \n\n' + report)

    elif method_name == "nn":

        # summarize performance of the classification
        print('\nThe overall accuracy score of neural network classifier is: ' +
              str(accuracy_score(y_test, predictions)))

        print('The overall f1 score of neural network classifier is: ' + str(f1_score(y_test, predictions)))

        print('The overall roc_auc score of neural network classifier is: ' + str(roc_auc_score(y_test, predictions)))

        report = classification_report(y_test, predictions, target_names=['0', '1'])
        print('\n\nA detailed classification report for support vector classifier: \n\n' + report)

# show accuracy
show_accuracy_of_methods("nn")
