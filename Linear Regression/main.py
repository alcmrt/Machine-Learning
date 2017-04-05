"""
Main file to run mini-batch gradient descent algorithm for both data sets.

RUN THIS FILE TO EXECUTE THE PROGRAM.
"""
from hypothesis import setParameters

__author__ = "Alicem Murat ÖCAL"
__email__ = "a.muratocal@gmail.com"


def main():
    """
    Main function to enter parameters and execute linear regression algorithm.
    """

    batch_size = int(input("Enter Batch Size: "))
    alpha = float(input("Enter learning rate: "))
    result_parameter = None

    print()
    print("Select data set(1 or 2) to apply mini-batch gradient descent algorithm.")
    print()
    print("The first data set is in \"student-por2.csv\" file")
    print("The second data set is in \"default_plus_chromatic_features_1059_tracks.txt\" file.")
    print()

    data_set = int(input("Now select the data set(1 or 2): "))

    # set result parameter
    if(data_set == 1):
        result_parameter = input("Enter result parameter(\"g1\", \"g2\" or \"g3\")"
                                 " in order to estimate: ")
    elif(data_set == 2):

        result_parameter = input("Enter result parameter(\"latitude\" or \"longitude\")"
                                 " in order to estimate: ")

    cost_function = input("Select name of the cost function(\"mse\" or \"mae\"): ")
    use_momentum = input("Do you want to use momentum(y/n): ")

    iteration_number = int(input("How many times do you want mini-batch gradient "
                                 "descent algorithm to run?(iteration number): "))

    iteration_number += 1

    print()
    print("Please wait ...")

    if use_momentum == "y":
        use_momentum = True
    else:
        use_momentum = False

    setParameters(alpha, batch_size, data_set, result_parameter, cost_function, use_momentum,iteration_number)


# execute the program
main()
