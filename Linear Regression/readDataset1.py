"""
For getting the data from "student-por2.csv" file
and separating them as test_data_matrix, training_data_matrix,
test_g1_list, test_g2_list, test_g3_list, training_g1_list,
training_g2_list and training_g3_list.

The first 20% of the data will be used as test data, the remaining
part will be training data.
"""

import csv

__author__ = "Alicem Murat ÖCAL"
__email__ = "a.muratocal@gmail.com"

line = None                         # read line

data_matrix = None                  # the data read from the file, stores all data in the file
training_data_matrix = None         # training data
test_data_matrix = None             # test data(the first 20% of the data)

# training result list
training_g1_list = None
training_g2_list = None
training_g3_list = None

# test result list
test_g1_list = None
test_g2_list = None
test_g3_list = None

row_number_of_data_matrix = 649    # number of rows of data_matrix
column_number_of_data_matrix = 52  # number of columns of data_matrix

single_row = None                   # single row of a data will be used as float list
vector_X = None                     # elements of predictors

# calculating the first 20% of the data
row_number_of_test_data_matrix = int(row_number_of_data_matrix * 0.2)

# create empty training_data_matrix
training_data_matrix = [[0.0] for i in range(row_number_of_data_matrix - row_number_of_test_data_matrix)]

# create empty test_data_matrix
test_data_matrix = [[0.0] for i in range(row_number_of_test_data_matrix)]

# create empty training g1, g2 and g3 lists.
training_g1_list = [[0.0] for i in range(row_number_of_data_matrix - row_number_of_test_data_matrix)]
training_g2_list = [[0.0] for i in range(row_number_of_data_matrix - row_number_of_test_data_matrix)]
training_g3_list = [[0.0] for i in range(row_number_of_data_matrix - row_number_of_test_data_matrix)]


# create empty test g1, g2 and g3 lists.(y1, y2 and y3 values in the data set.)
test_g1_list = [[0.0] for i in range(row_number_of_test_data_matrix)]
test_g2_list = [[0.0] for i in range(row_number_of_test_data_matrix)]
test_g3_list = [[0.0] for i in range(row_number_of_test_data_matrix)]


def open_file(file_path):
    """
    open csv file and read data.
    :param file_path: file path of the csv file that will be read.

    :return
    Test data matrix, training data matrix, training g1,g2,g3 and
    test g1, g2, g3 lists
    """

    global vector_X
    global training_data_matrix
    global row_number_of_data_matrix
    global single_row

    global training_g1_list
    global training_g2_list
    global training_g3_list

    global test_g1_list
    global test_g2_list
    global test_g3_list

    # open file
    with open(file_path, "r") as csvfile:

        line_number = 0
        index_of_training_matrix = 0

        # read all rows of csv file
        reader = csv.reader(csvfile)

        next(reader, None)  # skip the headers

        for row in reader:

            row = row[0]

            # read line split by comma and convert into float numbers
            single_row = [float(x) for x in row.split(";")]

            # take the first 20% of the data as test data
            # and the remaining as the training data
            if line_number < row_number_of_test_data_matrix:

                test_data_matrix[line_number] = [1.0] + single_row[:-3]

                test_g1_list[line_number] = single_row[-3]
                test_g2_list[line_number] = single_row[-2]
                test_g3_list[line_number] = single_row[-1]

            else:
                training_data_matrix[index_of_training_matrix] = [1.0] + single_row[:-3]

                training_g1_list[index_of_training_matrix] = single_row[-3]
                training_g2_list[index_of_training_matrix] = single_row[-2]
                training_g3_list[index_of_training_matrix] = single_row[-1]

                index_of_training_matrix += 1

            if line_number == (row_number_of_data_matrix - 1):
                break

            line_number += 1

    return test_data_matrix, training_data_matrix, \
           test_g1_list, test_g2_list, test_g3_list, \
           training_g1_list, training_g2_list, training_g3_list