import os
import sys
import numpy as np
from decision_tree import ClassificationTree, RegressionTree
from utils import get_random_subsets


class RandomForest(object):
    def __init__(self, n_estimators, max_features=None, max_depth=float("inf"),
            min_split_samples=2, min_impurity=1e-7, regression=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_split_samples = min_split_samples
        self.regression = regression

        if self.regression:
            DecisionTree = RegressionTree
        else:
            DecisionTree = ClassificationTree

        self.trees = []
        for _ in range(n_estimators):
            tree = DecisionTree(max_depth=max_depth,
                    min_split_samples=min_split_samples,
                    min_impurity=min_impurity)
            self.trees.append(tree)

    def fit(self, X, y):
        n_features = X.shape[1]

        if not self.max_features:
            self.max_features = int(np.sqrt(n_features))

        subsets = get_random_subsets(X, y, self.n_estimators)

        for i in range(self.n_estimators):
            X_subset, y_subset = subsets[i]
            feat_idx = np.random.choice(range(n_features), self.max_features, replace=True)
            self.trees[i].feat_indices = feat_idx
            X_subset = X_subset[:, feat_idx]
            self.trees[i].fit(X_subset, y_subset)

    def predict(self, X):
        y_preds = np.empty((X.shape[0], self.n_estimators))
        for i, tree in enumerate(self.trees):
            feat_idx = tree.feat_indices
            y_preds[:, i] = tree.predict(X[:, feat_idx])

        if self.regression:
            y_pred = np.mean(y_preds, axis=1)
        else:
            y_pred = []
            for sample_preds in y_preds:
                y_pred.append(np.bincount(sample_preds.astype('int')).argmax())
        return y_pred


class RandomForestClassifier(RandomForest):
    def __init__(self, n_estimators=100, max_features=None, max_depth=float("inf"),
            min_split_samples=2, min_impurity=1e-7):
        super().__init__(n_estimators=n_estimators,
                max_features=max_features,
                max_depth=max_depth,
                min_split_samples=min_split_samples,
                min_impurity=min_impurity,
                regression=False)



class RandomForestRegression(RandomForest):
    def __init__(self, n_estimators=100, max_features=None, max_depth=float("inf"),
            min_split_samples=2, min_impurity=1e-7):
        super().__init__(n_estimators=n_estimators,
                max_features=max_features,
                max_depth=max_depth,
                min_split_samples=min_split_samples,
                min_impurity=min_impurity,
                regression=True)



