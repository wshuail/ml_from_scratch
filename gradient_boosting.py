import os
import sys
import numpy as np

from decision_tree import RegressionTree

def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


class SquareLoss(object):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)



class CrossEntropy(object):
    def __init__(self):
        pass

    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -y*np.log(p) - (1-y)*(1-np.log(1-p))

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y/p) + (1-y)/(1-p)



class GradientBoosting(object):
    def __init__(self, n_estimators, learning_rate, max_depth=2,
            min_split_samples=2, min_impurity=1e-7, regression=False):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.regression = regression

        if self.regression:
            self.loss = SquareLoss()
        else:
            self.loss = CrossEntropy()
        
        self.trees = []
        for _ in range(n_estimators):
            tree = RegressionTree(max_depth=max_depth,
                    min_split_samples=min_split_samples,
                    min_impurity=min_impurity)
            self.trees.append(tree)

    def fit(self, X, y):
        y_preds = np.full(np.shape(y), np.mean(y, axis=0))
        for i in range(self.n_estimators):
            gradient = self.loss.gradient(y, y_preds)
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)
            # print (np.multiply(self.learning_rate, update).shape)
            y_preds -= np.multiply(self.learning_rate, update)
    
    def predict(self, X):
        y_preds = np.array([])
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_preds = -update if not y_preds.any() else y_preds - update

        if not self.regression:
            y_preds_exp = np.exp(y_preds)+1e-7
            y_preds_exp = np.reshape(y_preds_exp, (y_preds_exp.shape[0], -1))
            y_preds = y_preds_exp / (np.expand_dims(np.sum(y_preds_exp, axis=1), axis=1) + 1e-7)
            y_preds = np.argmax(y_preds, axis=1)
        return y_preds


class GradientBoostingRegression(GradientBoosting):
    def __init__(self, n_estimators=100, max_depth=4, learning_rate=0.5,
            min_split_samples=2, min_impurity=1e-7):
        super(GradientBoostingRegression, self).__init__(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_split_samples=min_split_samples,
                min_impurity=min_impurity,
                regression=True)


class GradientBoostingClassifier(GradientBoosting):
    def __init__(self, n_estimators=100, max_depth=4, learning_rate=0.5,
            min_split_samples=2, min_impurity=1e-7):
        super(GradientBoostingClassifier, self).__init__(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_split_samples=min_split_samples,
                min_impurity=min_impurity,
                regression=False)

    def fit(self, X, y):
        y = to_categorical(y)
        super(GradientBoostingClassifier, self).fit(X, y)


if __name__ == '__main__':
    from sklearn import datasets
    from utils import split_train_test, cal_accuracy
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







