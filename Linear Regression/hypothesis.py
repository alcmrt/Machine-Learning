"""
Use Linear Regression learning model with mini batch gradient descent algorithm
on data sets in "default_plus_chromatic_features_1059_tracks.txt" and
"student-por2.csv" files and show Cost-Iteration plot for the test data.
"""

from readDataset2 import openFile
from readDataset1 import open_file
import matplotlib.pyplot as plt
import decimal as d

__author__ = "Alicem Murat ÖCAL"
__email__ = "a.muratocal@gmail.com"

# Global variables
coefficient_vector = None  # list of coefficients for linear regression.
# vector_X = None                 # feature vector
number_of_coefficients = None  # number of coefficients

# how many times the batch-gradient descent algorithm will work
number_of_iteration = None

# learning rate for mini-batch gradient descent algorithm
alpha = 0.001

# batch size for mini-batch gradient descent algorithm
batchSize = 4

# default momentum coefficient for mini-batch gradient descent
sigma = 0.9


def calculate_result_of_hypothesis(vector_x):
    """
    calculate result of linear regression
    vector_x is our vector of features

    :param vector_x: an instance vector
    :return result of linear regression equation
    """

    global coefficient_vector

    # calculate result
    result = sum([coefficient * feature for coefficient, feature in zip(coefficient_vector, vector_x)])

    # return result of linear regression equation
    return result


def compute_mse_cost(data_matrix, real_result_list):
    """
    compute Mean Squared Error.

    :param data_matrix: contains test data or training data
    :param real_result_list: contains test or training results for latitude or longitude lists

    :return cost for MSE.
    """

    # get length of data_matrix
    n = len(data_matrix)

    # (n)-element result list of linear regression equation
    result_list_of_linear_regression = [0.0 for i in range(n)]

    # calculate linear regression results for all feature vectors
    # in data_matrix
    for index in range(0, n):
        # get feature vector from data_matrix
        vector_x = data_matrix[index]

        # calculate result of linear regression for all feature vector_x's in data_matrix
        result_list_of_linear_regression[index] = calculate_result_of_hypothesis(vector_x)

    # Python's decimal library is being used to prevent
    # Overflow 'result too large' error
    cost = d.Decimal(0)
    cost = d.Decimal(1 / n) * d.Decimal(sum([pow((d.Decimal(real_result_list[i]) - d.Decimal(result_list_of_linear_regression[i])), 2)
                        for i in range(0, n)]))

    return cost


def compute_mae_cost(data_matrix, real_result_list):
    """
    Compute Mean Absolute Error.

    :param data_matrix: contains test data or training data
    :param real_result_list: contains test or training results for latitude or longitude lists

    :return cost for MAE.
    """

    # get length of data_matrix
    n = len(data_matrix)

    # (n)-element result list of linear regression equation
    result_list_of_linear_regression = [0.0 for i in range(n)]

    # calculate linear regression results for all feature vectors
    # in data_matrix
    for index in range(0, n):
        # get feature vector from data_matrix
        vector_x = data_matrix[index]

        # calculate result of linear regression for all feature vector_x's in data_matrix
        result_list_of_linear_regression[index] = calculate_result_of_hypothesis(vector_x)

    # calculate cost for MAE cost function
    cost = 1 / n * sum([(abs(real_result_list[i] - result_list_of_linear_regression[i])) for i in range(0, n)])

    return cost


def mini_batch_gradient_descent1(training_data_matrix, result_list_of_training_data, test_data_matrix,
                                 result_list_of_test_data, type_of_cost_function="mse", use_momentum=False):
    """
    Mini-Batch Gradient Descent Algorithm for MSE and MAE cost functions.

    :param training_data_matrix: takes the training data matrix.
    :param result_list_of_training_data: takes the real result list(latitude and longitude) of features.
    :param test_data_matrix: takes the test data matrix to calculate error.
    :param result_list_of_test_data: takes the real result list(latitude longitude) of test data.
    :param type_of_cost_function: select the type of the cost function, "mse" stands for MEAN SQUARED ERROR and
                                  "mae" stands for MEAN ABSOLUTE ERROR. Default value is "mse".
    :param use_momentum: use momentum in the mini-batch gradient descent algorithm. Default value is False.
    """

    global alpha  # learning rate
    global batchSize  # batch Size

    global temp_coefficients
    global coefficient_vector

    global sigma  # momentum constant = 0.9

    global number_of_iteration

    # if batchSize > length of remaining subset
    # use this temporary_batch_size variable
    temporary_batch_size = batchSize

    # velocity given by momentum
    velocity = [0.0 for i in range(number_of_coefficients)]

    # history of the cost function will be stored here
    cost_history = []

    # iteration counter for mini-batch gradient descent
    iteration_counter = 1

    # indices for getting subset of data_matrix
    subset_from = 0
    subset_to = temporary_batch_size  # subset_to = 8

    # run mini batch gradient descent algorithm.
    while True:

        # stop gradient descent after completing number_of_iteration
        if iteration_counter == number_of_iteration:
            break

        if len(training_data_matrix) <= subset_from:
            subset_from = 0
            subset_to = batchSize
            temporary_batch_size = batchSize

        # out of range control
        elif subset_to > len(training_data_matrix):
            subset_to = len(training_data_matrix)
            temporary_batch_size = subset_to - subset_from

        # 8(batchSize) element result list of linear regression equation
        result_of_linear_regression = [0.0 for i in range(temporary_batch_size)]

        # get subset of data_matrix length of the batchSize
        subset_training_data_matrix = training_data_matrix[subset_from: subset_to]

        # get subset of result data length of the batchSize
        subset_result_list = result_list_of_training_data[subset_from: subset_to]

        # calculate (batchSize)8-element result list of linear regression
        for index in range(0, temporary_batch_size):  # range(0, 8)

            # get feature vector
            vector_x = subset_training_data_matrix[index]

            # calculate results of linear regression equation
            result_of_linear_regression[index] = calculate_result_of_hypothesis(vector_x)

        # if type of cost function is "mse", apply mini-batch gradient descent algorithm
        # for mean squared error.
        if type_of_cost_function == "mse":

            # run mini-batch gradient descent algorithm for "mse" cost function to calculate all coefficients
            for j in range(0, number_of_coefficients):  # range(0, 117)

                # momentum method is being used in mini-batch gradient descent algorithm.
                if(use_momentum):

                    # estimate coefficients by using mini-batch gradient descent algorithm with momentum method
                    velocity[j] = sigma * velocity[j] - \
                                  alpha / temporary_batch_size * \
                                  sum([((result_of_linear_regression[i] - subset_result_list[i]) *
                                        subset_training_data_matrix[i][j]) for i in range(0, temporary_batch_size)])

                    temp_coefficients[j] = coefficient_vector[j] + velocity[j]

                # momentum method is not being used.
                else:
                    # estimate coefficients by using mini-batch gradient descent algorithm
                    temp_coefficients[j] = coefficient_vector[j] - alpha / temporary_batch_size * \
                                                                   sum([((result_of_linear_regression[i] -
                                                                          subset_result_list[i]) *
                                                                         subset_training_data_matrix[i][j]) for i in
                                                                        range(0, temporary_batch_size)])

        # if type of cost function is "mae", apply mini-batch gradient descent algorithm
        # for mean absolute error.
        elif type_of_cost_function == "mae":

            # run mini-batch gradient descent algorithm for "mae" cost function to calculate all coefficients
            for j in range(0, number_of_coefficients):  # range(0, 117)

                # momentum method is being used in mini-batch gradient descent algorithm.
                if(use_momentum):

                    # estimate coefficients by using mini-batch gradient descent algorithm with momentum method
                    velocity[j] = sigma * velocity[j] - \
                                  alpha / temporary_batch_size * \
                                  sum([((result_of_linear_regression[i] - subset_result_list[i]) *
                                        subset_training_data_matrix[i][j]) for i in range(0, temporary_batch_size)])

                    temp_coefficients[j] = coefficient_vector[j] + velocity[j]

                else:
                    # estimate coefficients by using mini-batch gradient descent algorithm
                    temp_coefficients[j] = coefficient_vector[j] + alpha / temporary_batch_size * \
                                                                   sum([(subset_training_data_matrix[i][j]
                                                                         / abs(
                                                                       subset_result_list[i] - result_of_linear_regression[
                                                                           i]))
                                                                        for i in range(0, temporary_batch_size)])

        # update values of the coefficients
        for j in range(0, number_of_coefficients):  # range(0, 117)
            coefficient_vector[j] = temp_coefficients[j]

        # shift to next subset
        subset_from = subset_to
        subset_to += batchSize

        # if type_of_cost_function == "mse", calculate cost for
        # Mean Squared Error cost function
        if type_of_cost_function == "mse":

            # compute cost for test data
            cost = compute_mse_cost(test_data_matrix, result_list_of_test_data)

            # add result of the cost per iteration into cost_history list
            cost_history.append((iteration_counter, cost))

        # if type_of_cost_function == "mae", calculate cost for
        # Mean Absolute Error cost function
        elif type_of_cost_function == "mae":

            # compute cost for test data
            cost = compute_mae_cost(test_data_matrix, result_list_of_test_data)

            # add result of the cost per iteration into cost_history list
            cost_history.append((iteration_counter, cost))

        # increment iteration counter
        iteration_counter += 1

    return coefficient_vector, cost_history


def setParameters(learning_rate = 0.001, batch_size = 8, data_set = 1, result_parameter = None,
                  cost_function = "mse", use_momentum = False, iteration_number = 1001):
    """
    Set parameters to apply mini-batch gradient descent algorithm and
    execute mini-batch gradient descent

    :param learning_rate: learning rate of mini-batch gradient descent.
                          default value is 0.001.

    :param batch_size: batch size of mini-batch gradient descent.Default value
                       is 8.

    :param data_set: data set parametet. Choose 1 to select student data,
                     or select 2 in order to select "default_plus_chromatic_features"

    :param result_parameter: result parameter needed to be estimated.
                             (Ex: "latitude", "longitude" or "g1", "g2", "g3")

    :param cost_function: select cost function to apply("mse" or "mae")

    :param use_momentum: use momentum parameter(True or False)

    :param iteration_number: number of how many times mini-batch gradient descent algorithm
                             will be executed. default value is 1001
    """

    global alpha
    global batchSize
    global coefficient_vector
    global temp_coefficients
    global number_of_coefficients
    global number_of_iteration

    cost_history = []

    alpha = learning_rate # set learning rate
    batchSize = batch_size
    number_of_iteration = iteration_number

    ###########
    # set plot parameter
    if cost_function == "mse":
        plt.ylabel("MSE")
        plt.title("MSE Cost - Iteration Plot")

    if cost_function == "mae":
        plt.ylabel("MAE")
        plt.title("MAE Cost - Iteration Plot")

    plt.xlabel("Iteration")
    ###########


    # extract data of the first data set.
    if data_set == 1:

        test_data_matrix, training_data_matrix, \
        test_g1_list, test_g2_list, test_g3_list, \
        training_g1_list, training_g2_list, training_g3_list = open_file("data/student-por2.csv")

    # extract data of the second data set.
    elif data_set == 2:

        # get all data from our data set
        test_data_matrix, training_data_matrix, \
        test_latitude_list, test_longitude_list, \
        training_latitude_list, training_longitude_list \
            = openFile("data/default_plus_chromatic_features_1059_tracks.txt")

    # set number of coefficients for linear regression
    number_of_coefficients = len(training_data_matrix[0])

    # initialize vectors
    # coefficients of linear regression, zero(0) for initial coefficient values
    coefficient_vector = [0.0 for i in range(number_of_coefficients)]

    # temporary coefficients (will be used to calculate coefficients)
    temp_coefficients = [0.0 for i in range(number_of_coefficients)]

    # apply gradient descent on the first data set.
    if data_set == 1:
        if result_parameter == "g1":

            coefficient_list, cost_history = mini_batch_gradient_descent1(training_data_matrix, training_g1_list,
                                                                          test_data_matrix, test_g1_list,
                                                                          cost_function, use_momentum)

        elif result_parameter == "g2":

            coefficient_list, cost_history = mini_batch_gradient_descent1(training_data_matrix, training_g2_list,
                                                                          test_data_matrix, test_g2_list,
                                                                          cost_function, use_momentum)

        elif result_parameter == "g3":

            coefficient_list, cost_history = mini_batch_gradient_descent1(training_data_matrix, training_g3_list,
                                                                          test_data_matrix, test_g3_list,
                                                                          cost_function, use_momentum)

    # apply gradient descent on the second data set.
    elif data_set == 2:
        if result_parameter == "latitude":
            coefficient_list, cost_history = mini_batch_gradient_descent1(training_data_matrix, training_latitude_list,
                                                                          test_data_matrix, test_latitude_list,
                                                                          cost_function, use_momentum)

        elif result_parameter == "longitude":
            coefficient_list, cost_history = mini_batch_gradient_descent1(training_data_matrix, training_longitude_list,
                                                                          test_data_matrix, test_longitude_list,
                                                                          cost_function, use_momentum)

    # draw a plot for cost and iteration
    x_val = [x[0] for x in cost_history]
    y_val = [x[1] for x in cost_history]

    plt.plot(x_val, y_val)
    plt.show()
