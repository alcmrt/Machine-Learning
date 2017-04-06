"""
K - Nearest Neighbor Classifier.
"""
import csv
import pandas as pd
import math
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from collections import Counter

__author__ = "Alicem Murat ÖCAL"

##########################################################################
"""
Global variables
"""
# predict category of test data and get a list of predictions
predictions = []

# category/class of predictions
prediction_categories = []

# probabilities of predictions
prediction_probabilities = []
##########################################################################

# read data
data = pd.read_csv("data/train.csv", sep=",")

# split data set as x and y
y = data.iloc[:, 0]
X = data.iloc[:, 1:]

# normalize features by using mean normalization
for i in X:
    X[i] = (X[i] - X[i].mean())/X[i].std()


# feature selection, get 200 best features applying by F-Test score.
X = SelectKBest(score_func=f_classif, k=200).fit_transform(X, y)

# split data as training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# convert pandas data frame to numpy.ndarray
y_train = y_train.values
y_test = y_test.values


def calculate_distance1(data1, data2):
    """
    :param data1: a vector
    :param data2: a vector

    :return:
     euclidean distance between data1 and data2
    """

    # zip data1 and data2
    zipped = zip(data1, data2)

    # calculate distance between data1 and data2
    distance = math.sqrt(sum([pow(a - b, 2) for (a, b) in zipped]))

    return distance


def get_neighbors(X_train, test_sample, k):

    """
    :param X_train: list of training samples
    :param test_sample: a single test sample
    :param k : number of neighbors

    :return: training index list of k nearest neighbors of given test sample
    """

    # empty distance list
    distance_list = []

    # training index list of k-nearest neighbors
    neighbors = []  # this is not being used

    # calculate distance between test_sample and all training_samples
    for training_index in range(0, len(X_train)):

        training_sample = X_train[training_index]

        # calculate distance between test_sample and training_sample
        distance = calculate_distance1(test_sample, training_sample)

        # append training index and distance into distance_list
        distance_list.append((training_index, distance))

    # sort distance_list, from smallest distance to biggest
    distance_list.sort(key=lambda tup: tup[1])

    # get training index list of k-nearest neighbors
    #neighbors = [x[0] for x in distance_list[0: k]]

    # get class list of 5 nearest training neighbors
    #class_list_of_nearest_neighbors = [y_train[i] for i in neighbors]
    class_list_of_nearest_neighbors = [y_train[x[0]] for x in distance_list[0: k]]

    # return class list of nearest neighbors
    return class_list_of_nearest_neighbors


def detect_majority(class_list_of_nearest_neighbors):
    """
    calculate probability and majority of a class

    :param class_list_of_nearest_neighbors: list of classes of nearest neighbors
    :return major category in classes_of_nearest_neighbors
    """

    # count categories
    count = Counter(class_list_of_nearest_neighbors)

    # get probability of "class 1"
    probability_of_class1 = 1/len(class_list_of_nearest_neighbors) * class_list_of_nearest_neighbors.count(1)

    # return majority and probability of the major category
    return count.most_common()[0][0], probability_of_class1


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
    data_matrix = [[0.0] for j in range(row_number_of_data + 1)]

    data_matrix[0] = ["real test categories", "estimated probabilities"]

    # write real categories of test data and estimated class probabilities
    # of test data into data_matrix
    for j in range(1, row_number_of_data + 1):
        data_matrix[j] = [y_test[j-1]] + [prediction_probabilities[j-1]]

    # write results into a csv file.
    write_file(file_path="data/results_of_knn.csv", data_matrix=data_matrix)

#########################################################################################

# list of predictions
predictions = []

# estimate categories of every sample in test data by using KNN Classifier.
for test_index in range(0, len(X_test)):

    print('Classifying test instance number ' + str(test_index) + ":",)

    # get class list of 3 nearest training neighbors
    classes_of_nearest_neighbors = get_neighbors(X_train, X_test[test_index], 3)

    # get major class of nearest neighbors and probability of "class 1"
    major_class, category_1_probability = detect_majority(classes_of_nearest_neighbors)

    # store predictions in predictions list for evaluate accuracy
    predictions.append((major_class, category_1_probability))

    # show estimations for every test sample
    print("Probability of class 1=" + str(category_1_probability) + ", Predicted label=" + str(major_class) +
          ", Actual Label=" + str(y_test[test_index]))
    print()

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
print('\nThe overall accuracy score of knn is: ' + str(accuracy_score(y_test, prediction_categories)))
print('\nThe overall f1 score of knn is: ' + str(f1_score(y_test, prediction_categories)))
print('\nThe overall roc_auc score of knn is: ' + str(roc_auc_score(y_test, prediction_categories)))

# create a classification report by using sklearn.
report = classification_report(y_test, prediction_categories, target_names=['0', '1'])
print('\n\nA detailed classification report: \n\n' + report)
