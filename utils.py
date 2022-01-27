import os
import sys
import numpy as np
from sklearn import datasets


def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot



def shuffle_data(X, y):
    idx = list(range(X.shape[0]))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    return X, y


def split_train_test(X, y, ratio=0.3):
    train_size = int(X.shape[0]*(1-ratio))
    X, y = shuffle_data(X, y)
    train_X, test_X = X[: train_size], X[train_size: ]
    train_y, test_y = y[: train_size], y[train_size: ]
    return train_X, train_y, test_X, test_y


def devide_on_feat(X, feat_i, threshold):
    split_func = lambda sample: sample[feat_i]>=threshold
    left = [sample for sample in X if split_func(sample)]
    right = [sample for sample in X if not split_func(sample)]
    left, right = np.array(left), np.array(right)
    return left, right


def cal_entropy(y):
    log2 = lambda x: np.log(x)/np.log(2)
    entropy = 0
    for unique_value in np.unique(y):
        p = np.sum(y==unique_value)/len(y)
        entropy += -p*log2(p)
    return entropy

def mean_of_y(y):
    value = np.mean(y, axis=0)
    return value if len(value) > 1 else value[0]

def cal_accuracy(y, preds):
    return np.sum(y==preds)/len(y)

def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred """
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse

def calculate_variance(X):
    """ Return the variance of the features in dataset X """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
    
    return variance

def standardize(X):
    """ Standardize the dataset X """
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std




