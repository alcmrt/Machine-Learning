"""
Parameter optimization using grid search with cross-validation
via scikit-learn.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import neural_network
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

__author__ = "Alicem Murat ÖCAL"


def optimize_parameters(method_name):
    """
    :param method_name: decide method for parameter optimization
                    'log_reg' for logistic regression,
                    'svm' for support vector machine,
                    'nn' for neural network,
                    'ridge' for ridge regression,
                    'knn' for k-nearest neighbors classifier
                    'dt' for decision tree
    """

    # read data
    data = pd.read_csv("data/train.csv", sep=",")

    # split data set as x and y
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]

    # normalize features by using mean normalization
    for i in X:
        X[i] = (X[i] - X[i].mean())/X[i].std()

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    # optimize parameters for support vector machine
    if method_name == 'svm':

        # Set the parameters by cross-validation (for Support Vector Machine)
        tuned_parameters = [{'kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
                             'gamma': [1e-1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000],
                             'degree': [2, 3, 4]}]

        #scores = ['precision', 'recall']
        #scores = ['f1']
        scores = ['accuracy']
        #scores = ['average_precision']

        for score in scores:
            print()
            print("PLEASE WAIT")
            print()
            print("# Tuning parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(), tuned_parameters, cv=10, scoring='%s' % score)
            clf.fit(X_train, y_train)

            print("--------------------------------------------------------")
            print("FOR SUPPORT VECTOR MACHINE")
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("--------------------------------------------------------")
            print("Grid scores on development set:")
            print()

            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()

    # optimize parameters for logistic regression
    elif method_name == 'log_reg':

        # Set the parameters by cross-validation (for Support Vector Machine)
        tuned_parameters = [{'solver': ['newton-cg'], 'max_iter': [1000]},
                            {'solver': ['lbfgs'], 'max_iter': [1000]},
                            {'solver': ['sag'], 'max_iter': [1000]}]

        scores = ['accuracy']

        for score in scores:
            print()
            print("PLEASE WAIT")
            print()
            print("# Tuning parameters for %s" % score)
            print()

            clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=10, scoring='%s' % score)
            clf.fit(X_train, y_train)

            print("--------------------------------------------------------")
            print("FOR LOGISTIC REGRESSION")
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("--------------------------------------------------------")
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()

    # optimize parameters for ridge regression
    elif method_name == 'ridge':

        # Set the parameters by cross-validation (for Support Vector Machine)
        tuned_parameters = [{'alpha': [1, 10, 100, 1000], 'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
                             'max_iter': [1000]}]

        scores = ['accuracy']

        for score in scores:
            print()
            print("PLEASE WAIT")
            print()
            print("# Tuning parameters for %s" % score)
            print()

            clf = GridSearchCV(RidgeClassifier(), tuned_parameters, cv=10, scoring='%s' % score)
            clf.fit(X_train, y_train)

            print("--------------------------------------------------------")
            print("FOR RIDGE REGRESSION")
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("--------------------------------------------------------")
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()

    # optimize parameters for k nearest neighbor classifier
    elif method_name == 'knn':

        # Set the parameters by cross-validation (for Support Vector Machine)
        tuned_parameters = [{'n_neighbors': [5, 10], 'weights': ['uniform', 'distance']}]

        scores = ['accuracy']

        for score in scores:
            print()
            print("Please wait ...")
            print()
            print("# Tuning parameters for %s" % score)
            print()

            clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=10, scoring='%s' % score)
            clf.fit(X_train, y_train)

            print("--------------------------------------------------------")
            print("FOR KNN CLASSIFIER")
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("--------------------------------------------------------")
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()

    # optimize parameters for neural network
    elif method_name == 'dt':
        # Set the parameters by cross-validation (for Support Vector Machine)
        tuned_parameters = [{'criterion': ['entropy'],
                             'splitter': ['best', 'random'],
                             'max_features': ['auto', 'sqrt', 'log2'],
                             'presort': [True, False],
                             'max_depth':[4, 5, 6, 7, 8, 9, 10]}]

        scores = ['accuracy']

        for score in scores:
            print()
            print("Please Wait ...")
            print()
            print("# Tuning parameters for %s ..." % score)
            print()

            clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=10, scoring='%s' % score)
            clf.fit(X_train, y_train)

            print("--------------------------------------------------------")
            print("FOR Decision Tree")
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("--------------------------------------------------------")
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()

    # optimize parameters for neural network
    elif method_name == 'nn':
        # Set the parameters by cross-validation (for Support Vector Machine)
        tuned_parameters = [{'activation': ['relu'], 'solver': ['adam'], 'alpha': [0.1], 'batch_size': [10],
                             'learning_rate': ['constant'], 'max_iter': [900], 'momentum': [0.9]},
                            {'activation': ['relu'], 'solver': ['adam'], 'alpha': [0.01], 'batch_size': [10],
                             'learning_rate': ['constant'], 'max_iter': [900], 'momentum': [0.9]},
                            {'activation': ['relu'], 'solver': ['adam'], 'alpha': [0.001], 'batch_size': [10],
                             'learning_rate': ['constant'], 'max_iter': [900], 'momentum': [0.9]},

                            {'activation': ['identity'], 'solver': ['adam'], 'alpha': [0.1], 'batch_size': [10],
                             'learning_rate': ['constant'], 'max_iter': [900], 'momentum': [0.9]},
                            {'activation': ['identity'], 'solver': ['adam'], 'alpha': [0.01], 'batch_size': [10],
                             'learning_rate': ['constant'], 'max_iter': [900], 'momentum': [0.9]},
                            {'activation': ['identity'], 'solver': ['adam'], 'alpha': [0.001], 'batch_size': [10],
                             'learning_rate': ['constant'], 'max_iter': [900], 'momentum': [0.9]},

                            {'activation': ['tanh'], 'solver': ['adam'], 'alpha': [0.1], 'batch_size': [10],
                             'learning_rate': ['constant'], 'max_iter': [900], 'momentum': [0.9]},
                            {'activation': ['tanh'], 'solver': ['adam'], 'alpha': [0.01], 'batch_size': [10],
                             'learning_rate': ['constant'], 'max_iter': [900], 'momentum': [0.9]},
                            {'activation': ['tanh'], 'solver': ['adam'], 'alpha': [0.001], 'batch_size': [10],
                             'learning_rate': ['constant'], 'max_iter': [900], 'momentum': [0.9]},

                            {'activation': ['logistic'], 'solver': ['adam'], 'alpha': [0.1], 'batch_size': [10],
                             'learning_rate': ['constant'], 'max_iter': [900], 'momentum': [0.9]},
                            {'activation': ['logistic'], 'solver': ['adam'], 'alpha': [0.01], 'batch_size': [10],
                             'learning_rate': ['constant'], 'max_iter': [900], 'momentum': [0.9]},
                            {'activation': ['logistic'], 'solver': ['adam'], 'alpha': [0.001], 'batch_size': [10],
                             'learning_rate': ['constant'], 'max_iter': [900], 'momentum': [0.9]}]

        scores = ['accuracy']

        for score in scores:
            print()
            print("Please Wait ...")
            print()
            print("# Tuning parameters for %s ..." % score)
            print()

            clf = GridSearchCV(neural_network.MLPClassifier(), tuned_parameters, cv=10, scoring='%s' % score)
            clf.fit(X_train, y_train)

            print("--------------------------------------------------------")
            print("FOR NEURAL NETWORK")
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("--------------------------------------------------------")
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()


method = input("Enter method to optimize its parameters:")

if method == 'svm':  # for support vector machine
    optimize_parameters('svm')

elif method == 'log_reg':  # for logistic regression
    optimize_parameters('log_reg')

elif method == 'nn':  # for neural network
    optimize_parameters('nn')

elif method == 'ridge':  # for ridge regression
    optimize_parameters('ridge')

elif method == 'knn':  # for k nearest neighbors classifier
    optimize_parameters('knn')

elif method == 'dt': # for decision tree
    optimize_parameters('dt')

else:
    print("Unknown method name, please try again.")
