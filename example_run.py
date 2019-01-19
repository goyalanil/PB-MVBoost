"""
This code is an example for running code for  multiview learning algorithm based on boosting. We call this algorithm
as PB-MVBoost.

Related Paper:
Multiview Boosting by Controlling the Diversity and the Accuracy of View-specific Voters
by Anil Goyal, Emilie Morvant, Pascal Germain and Massih-Reza Amini

Link to the paper:
https://arxiv.org/abs/1808.05784
"""
__author__="Anil Goyal"
import pickle
from MVL.PB_MVBoost import PB_MVBoost

def read_data(path_to_data,views):
    """
    This function reads the data.
    :return:
    """

    X_train = {}
    y_train = {}
    X_test = {}
    y_test = {}

    for name_of_view in views:
        # view-specific training data
        X_train[name_of_view] = pickle.load(open(path_to_data + name_of_view + "/" +
                                                      name_of_view + "_X_train.p", "rb"), encoding='latin1')

        y_train[name_of_view] = pickle.load(open(path_to_data + name_of_view + "/" +
                                                      name_of_view + "_y_train.p", "rb"), encoding='latin1')

        # view-specific test data
        X_test[name_of_view] = pickle.load(open(path_to_data + name_of_view + "/" +
                                                     name_of_view + "_X_test.p", "rb"), encoding='latin1')
        y_test[name_of_view] = pickle.load(open(path_to_data + name_of_view + "/" +
                                                     name_of_view + "_y_test.p", "rb"), encoding='latin1')

    return X_train,y_train,X_test,y_test

def main():
    """
    An example for executing code for  PB-MVBoost.
    """
    #Initializing parameters
    num_iterations_PBMVboost=10
    tree_depth=2
    views = ['view1', 'view2', 'view3', 'view4']

    #Reading Data. Note that this is a sample data
    path_to_data = "sample_data/"
    X_train, y_train, X_test, y_test = read_data(path_to_data,views)

    #Creating an Object for running algorithm
    pbmvboost_obj = PB_MVBoost(X_train,y_train, X_test, y_test, views, num_iterations=num_iterations_PBMVboost,
                            decision_tree_depth=tree_depth)
    #Learning Model
    pbmvboost_obj.learn()


if __name__ == '__main__':
    main()
