"""
Logistic Regression Classifier with
batch gradient descent algorithm.

Splits data set as test and training data.
Shows Cost-Iteration plot for the test data
and accuracy scores for different performance metrics.
"""

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

__author__ = "Alicem Murat ÖCAL"

# read data
data = pd.read_csv("data/train.csv", sep=",")

# split data set as X and y
y = data.iloc[:, 0]
X = data.iloc[:, 1:]


# normalize features by using mean normalization
for i in X:
    X[i] = (X[i] - X[i].mean())/X[i].std()


# feature selection, get 250 best features applying by F-Test score.
X = SelectKBest(score_func=f_classif, k=250).fit_transform(X, y)

# Add Bias term
X = pd.DataFrame(np.hstack((np.ones((X.shape[0], 1)), X)))

# convert pandas data frame to numpy.ndarray
X = X.values

# split data as training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# convert pandas data frame to numpy.ndarray
y_train = y_train.values
y_test = y_test.values


"""
Global variables
"""
# number of coefficients
number_of_coefficients = len(X_train[0])

# coefficients of logistic regression, zero(0.0) for initial coefficient values
coefficient_vector = [0.0 for i in range(number_of_coefficients)]

# temporary coefficients (will be used to calculate coefficients)
temp_coefficients = [0.0 for i in range(number_of_coefficients)]

##########################################################################
# predict category of test data and get a list of predictions
predictions = []

# category/class of predictions
prediction_categories = []

# probabilities of predictions
prediction_probabilities = []
##########################################################################


def sigmoid_function(x):
    """
    Calculate result of sigmoid function.

    :param x: input value for sigmoid function
    :return: result of sigmoid function.
    """
    result = 1.0 / (1 + math.e**(-x))

    return result


def calculate_result_of_logistic_regression_equation(vector_x):
    """
    Calculate result of logistic regression hypothesis.

    :param vector_x: a feature vector
    :return: estimation of logistic regression
    """

    global coefficient_vector

    # calculate result of equation
    result = sum([coefficient * feature for coefficient, feature in zip(coefficient_vector, vector_x)])

    # calculate result of sigmoid function
    result_of_sigmoid_function = sigmoid_function(result)

    # return result of logistic regression estimation
    return result_of_sigmoid_function


def estimate_category_of_test_data(test_data_matrix, coefficient_vector):
    """
    Estimate class/category of test data

    :param test_data_matrix: test data(X_test).
    :param coefficient_vector: coefficient vector found after applying batch gradient descent.

    :return: an estimation list
    """

    predictions = []  # list of predictions and their probabilities

    # estimate category for all test data
    for vector_x in test_data_matrix:

        # calculate result of equation
        result = sum([coefficient * feature for coefficient, feature in zip(coefficient_vector, vector_x)])

        # calculate result of sigmoid function
        result_of_sigmoid_function = sigmoid_function(result)

        # if estimation probability >= 0.5
        # category is 1 else category is 0.
        if result_of_sigmoid_function >= 0.5:
            predictions.append((1, result_of_sigmoid_function))
        else:
            predictions.append((0, result_of_sigmoid_function))

    # return a list of estimations.
    return predictions


def logistic_regression_cost(data_matrix, y):
    """
    Calculate error of logistic regression

    :param data_matrix: test data or training data
    :param y: result list of test data or training data

    :return: cost of logistic regression
    """

    # get length of data_matrix
    n = len(data_matrix)

    # (n)-element result list of logistic regression equation
    result_list_of_logistic_regression = [0.0 for i in range(n)]

    # calculate logistic regression results for all feature vectors
    # in data_matrix
    for index in range(0, n):

        # get feature vector from data_matrix
        vector_x = data_matrix[index]

        # calculate results of logistic regression equation and store it
        result_list_of_logistic_regression[index] = calculate_result_of_logistic_regression_equation(vector_x)

    # calculate cost of log loss error.
    cost = (-1.0 / n) * \
           sum([(y[i] * math.log(result_list_of_logistic_regression[i]) +
                 ((1.0 - y[i]) * math.log(1.0 - result_list_of_logistic_regression[i]))) for i in range(0, n)])

    # return cost of logistic regression
    return cost


def batch_gradient_descent(X_train, y_train, alpha=0.001, iteration_number=100, sigma=0.9):
    """
    Batch gradient descent for logistic regression.

    :param X_train: training data.
    :param y_train: results of training data
    :param alpha: learning rate, default value=0.001
    :param iteration_number: max number of iteration
    :param sigma: momentum constant

    :return: coefficient vector.
    """

    # iteration counter for mini-batch gradient descent
    iteration_counter = 0

    global number_of_coefficients
    global temp_coefficients
    global coefficient_vector

    # velocity given by momentum
    velocity = [0.0 for i in range(number_of_coefficients)]

    # history of the cost function will be stored here
    cost_history = []

    # calculate estimations of logistic regression and store it in that list
    results_of_logistic_regression = [0.0 for i in range(len(X_train))]

    while True:

        # stop gradient descent after completing number_of_iteration
        if iteration_counter == iteration_number:
            break

        # calculate results of logistic regression equation and
        # store the results in the list named as 'results_of_logistic_regression'.
        for index in range(0, len(X_train)):

            # get feature vector from training data
            vector_x = X_train[index]

            # calculate results of logistic regression equation and store it
            # in the list named results_of_logistic_regression.
            results_of_logistic_regression[index] = calculate_result_of_logistic_regression_equation(vector_x)

        # run batch gradient descent algorithm.(with momentum)
        for j in range(0, number_of_coefficients):

            # estimate coefficients by using batch gradient descent algorithm
            velocity[j] = sigma * velocity[j] - (alpha / len(X_train)) * \
                                                           sum([((results_of_logistic_regression[i] - y_train[i]) *
                                                                 X_train[i][j]) for i in range(0, len(X_train))])

            temp_coefficients[j] = coefficient_vector[j] + velocity[j]

        # update values of the coefficients
        for j in range(0, number_of_coefficients):
            coefficient_vector[j] = temp_coefficients[j]

        # increment iteration counter
        iteration_counter += 1

        # compute cost for test data
        cost = logistic_regression_cost(X_test, y_test)

        # add result of the cost per iteration into cost_history list
        cost_history.append((iteration_counter, cost))

    # return coefficients and cost history
    return coefficient_vector, cost_history


def write_file(file_path, data_matrix):
    """
    write data to csv file

    :param file_path: file path of data
    :param data_matrix: the data that will be written
    """

    # open file
    with open(file_path, mode="w", newline='') as file:

        writer = csv.writer(file)
        writer.writerows(data_matrix)


def create_data_matrix():
    """
    Create empty data_matrix to store estimated result probabilities of
    test data and the real test results of test data.
    """

    # create empty data_matrix to store estimated result probabilities of
    # test data and the real results
    row_number_of_data = len(X_test)
    data_matrix = [[0.0] for i in range(row_number_of_data + 1)]

    data_matrix[0] = ["real test categories", "estimated probabilities"]

    # write real categories of test data and estimated class probabilities
    # of test data into data_matrix
    for i in range(1, row_number_of_data + 1):
        data_matrix[i] = [y_test[i-1]] + [prediction_probabilities[i-1]]

    # write results into a csv file.
    write_file(file_path="data/results_of_logistic_regression.csv", data_matrix=data_matrix)


#######################################################################################

def run_logistic_regression():
    """
    Run logistic regression and show accuracy.

    Writes real test results and predicted predicted probabilities
    into results_of_logistic_regression.csv file.
    """

    global predictions
    global prediction_probabilities
    global prediction_categories

    alpha = 0.01  # learning rate.
    iteration_number = 88  # iteration number of batch gradient descent.
    sigma = 0.9  # momentum constant.

    print("\nRunning batch gradient descent algorithm on training data.")
    print("\nPlease wait ... \n")

    # get coefficient vector and cost history after training process
    coefficients, cost_history = batch_gradient_descent(X_train, y_train, alpha=alpha, iteration_number=iteration_number,
                                                        sigma=sigma)

    print("\nNow making estimation on the test data, please wait ...\n")

    # predict category of test data and get a list of predictions
    predictions = estimate_category_of_test_data(X_test, coefficients)

    # category/class of predictions
    prediction_categories = [x[0] for x in predictions]

    # probabilities of predictions
    prediction_probabilities = [x[1] for x in predictions]

    """
    Create empty data_matrix to store estimated result probabilities of
    test data and the real test results of test data.
    """
    create_data_matrix()

    # summarize performance of the classification
    print('\nThe overall accuracy score of logistic regression is: ' + str(accuracy_score(y_test, prediction_categories)) +
          "\n")

    print('\nThe overall f1 score of logistic regression is: ' + str(f1_score(y_test, prediction_categories)) + "\n")

    print('\nThe overall roc_auc score of logistic regression is: ' + str(roc_auc_score(y_test, prediction_categories))
          + "\n")

    report = classification_report(y_test, prediction_categories, target_names=['0', '1'])
    print('A detailed classification report for logistic regression: \n\n' + report)

    # set plot parameters
    plt.ylabel("Log Loss Error")
    plt.title("Logistic Regression Cost - Iteration Plot " + "for alpha=" + str(alpha) +
              ", itration=" + str(iteration_number) + ", sigma=" + str(sigma))

    plt.xlabel("Iteration")

    # draw a plot for cost and iteration
    x_val = [x[0] for x in cost_history]
    y_val = [x[1] for x in cost_history]

    plt.plot(x_val, y_val)
    plt.show()

#################################################################################################################

# run logistic regression
run_logistic_regression()
