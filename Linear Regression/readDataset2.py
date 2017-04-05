"""
For getting the data from "default_plus_chromatic_features_1059_tracks.txt" file
and separating them as test_data_matrix, training_data_matrix,
test_latitude_list, test_longitude_list, training_latitude_list
and training_longitude_list.

The first 20% of the data will be used as test data, the remaining
part will be training data.
"""

__author__ = "Alicem Murat ÖCAL"
__email__ = "a.muratocal@gmail.com"

# global variables
file = None                         # content of opened file
line = None                         # read line
data_matrix = None                  # the data read from the file, stores all data in the file
training_data_matrix = None         # training data
test_data_matrix = None             # test data(the first 20% of the data)

# training latitude and longitude lists
# y1 and y2 values in the data set
training_latitude_list = None       # training latitude data
training_longitude_list = None      # training longitude data

# test latitude and longitude lists
test_latitude_list = None           # test latitude data
test_longitude_list = None          # test longitude data

row_number_of_data_matrix = 1059    # number of rows of data_matrix
column_number_of_data_matrix = 118  # number of columns of data_matrix

single_row = None                   # single row of a data will be used as float list
vector_X = None                     # elements of predictors

# calculating the first 20% of the data
row_number_of_test_data_matrix = int(row_number_of_data_matrix * 0.2)

# create empty training_data_matrix
training_data_matrix = [[0.0] for i in range(row_number_of_data_matrix - row_number_of_test_data_matrix)]

# create empty training latitude and training longitude lists.
training_latitude_list = [[0.0] for i in range(row_number_of_data_matrix - row_number_of_test_data_matrix)]
training_longitude_list = [[0.0] for i in range(row_number_of_data_matrix - row_number_of_test_data_matrix)]

# create empty test_data_matrix
test_data_matrix = [[0.0] for i in range(row_number_of_test_data_matrix)]

# create empty test latitude and training longitude lists.
test_latitude_list = [[0.0] for i in range(row_number_of_test_data_matrix)]
test_longitude_list = [[0.0] for i in range(row_number_of_test_data_matrix)]


def openFile(filePath):
    """
    Open txt file and read data.
    :param filePath: file path of the txt file that will be read.

    :return
    Test data matrix, training data matrix, training latitude, longitude and
    test latitude, longitude lists
    """

    global vector_X
    global training_data_matrix
    global row_number_of_data_matrix
    global line
    global single_row

    global training_latitude_list
    global training_longitude_list
    global test_latitude_list
    global test_longitude_list

    # open file
    with open(filePath, mode="r") as file:

        lineNumber = 0
        index_of_training_matrix = 0

        # read all lines of text text file
        while True:
            # read line
            line = file.readline()

            # read line split by comma and convert into float numbers
            single_row = [float(x) for x in line.split(",")]

            # take the first 20% of the data as test data
            # and the remaining as the training data
            if(lineNumber < row_number_of_test_data_matrix):

                test_data_matrix[lineNumber] = [1.0] + single_row[:-2]
                test_latitude_list[lineNumber] = single_row[-2]
                test_longitude_list[lineNumber] = single_row[-1]
            else:
                training_data_matrix[index_of_training_matrix] = [1.0] + single_row[:-2]
                training_latitude_list[index_of_training_matrix] = single_row[-2]
                training_longitude_list[index_of_training_matrix] = single_row[-1]

                index_of_training_matrix += 1

            # if line doesn't have any data
            if not line:
                break
            elif lineNumber == (row_number_of_data_matrix - 1):
                break

            lineNumber += 1

    return test_data_matrix, training_data_matrix,\
           test_latitude_list, test_longitude_list,\
           training_latitude_list, training_longitude_list