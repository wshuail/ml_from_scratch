import os
import sys
import argparse
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from decision_tree import RegressionTree, ClassificationTree
from gradient_boosting import GradientBoostingClassifier, GradientBoostingRegression
from utils import split_train_test, cal_accuracy, mean_squared_error
from utils import standardize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    args = parser.parse_args()
    return args



def test_decision_tree_regression(gradient_boosting=True):

    print ("-- Regression Tree --")

    # Load temperature data
    data = pd.read_csv('data/TempLinkoping2016.txt', sep="\t")

    time = np.atleast_2d(data["time"].values).T
    temp = np.atleast_2d(data["temp"].values).T

    X = standardize(time)        # Time. Fraction of the year [0, 1]
    y = temp[:, 0]  # Temperature. Reduce to one-dim
    print (X.shape, y.shape)

    X_train, y_train, X_test, y_test = split_train_test(X, y)

    if gradient_boosting:
        model = GradientBoostingRegression()
    else:
        model = RegressionTree()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_line = model.predict(X)

    # Color map
    cmap = plt.get_cmap('viridis')

    mse = mean_squared_error(y_test, y_pred)

    print ("Mean Squared Error:", mse)

    # Plot the results
    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * X_test, y_pred, color='black', s=10)
    plt.suptitle("Regression Tree")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()


def test_decision_tree_classification():
    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target
    print (X.shape, y.shape)
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    print (train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    clf = ClassificationTree()
    clf.fit(train_X, train_y)
    preds = clf.predict(test_X)
    accuracy = cal_accuracy(test_y, preds)
    print (accuracy)


def test_gradient_boosting_classification():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    print (X.shape, y.shape)
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    print (train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    clf.fit(train_X, train_y)
    preds = clf.predict(test_X)
    accuracy = cal_accuracy(test_y, preds)
    print ('accuracy: ', accuracy)


models = {
        'decision_tree_regression': test_decision_tree_regression,
        'decision_tree_classification': test_decision_tree_classification,
        'gradient_boosting_classification': test_gradient_boosting_classification,
        'dtr': test_decision_tree_regression,
        'dtc': test_decision_tree_classification,
        'gbc': test_gradient_boosting_classification
        }






if __name__ == '__main__':
    args = parse_args()
    model = args.model
    models[model]()





