import os
import sys
import numpy as np
from sklearn import datasets

from utils import to_categorical, shuffle_data, split_train_test, devide_on_feat
from utils import cal_entropy, mean_of_y, cal_accuracy, mean_squared_error, calculate_variance
from utils import standardize


class DecisionNode(object):
    def __init__(self, feat_i=None, threshold=None, left_branch=None, right_branch=None, value=None):
        self.feat_i = feat_i
        self.threshold = threshold
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.value = value


class DecisionTree(object):
    def __init__(self, max_depth=float("inf"), min_split_samples=2, min_impurity=1e-7):
        self.max_depth = max_depth
        self.min_split_samples = min_split_samples
        self.min_impurity = min_impurity

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self.root

    def _build_tree(self, X, y, depth):
        largest_impurity = 0
        best_criteria = None
        best_sets = None

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)
        
        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)
        if n_samples > self.min_split_samples and depth <= self.max_depth:
            for i in range(n_features):
                feat_values = np.unique(Xy[:, i])
                for threshold in feat_values:
                    left_branch, right_branch = devide_on_feat(Xy, i, threshold)
                    if left_branch.shape[0] > 0 and right_branch.shape[0] > 0:
                        left_y, right_y = left_branch[:, n_features:], right_branch[:, n_features:]
                        impurity = self._impurity_calculation(y, left_y, right_y)
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {
                                    "feat_i": i,
                                    "threshold": threshold
                                    }
                            best_sets = {
                                    "left_X": left_branch[:, :n_features],
                                    "left_y": left_branch[:, n_features:],
                                    "right_X": right_branch[:, :n_features],
                                    "right_y": right_branch[:, n_features:]
                                    }
        if largest_impurity > self.min_impurity:
            left_branch = self._build_tree(best_sets['left_X'], best_sets['left_y'], depth+1)
            right_branch = self._build_tree(best_sets['right_X'], best_sets['right_y'], depth+1)
            return DecisionNode(feat_i=best_criteria['feat_i'], threshold=best_criteria['threshold'],
                    left_branch=left_branch, right_branch=right_branch)
        
        value = self._leaf_value_calculation(y)
        
        return DecisionNode(value=value)

    def predict(self, X):
        preds = [self.predict_sample(sample) for sample in X]
        return preds

    def predict_sample(self, sample, tree=None):
        if tree is None:
            tree = self.root

        if tree.value is not None:
            return tree.value

        feat_i = tree.feat_i
        threshold = tree.threshold
        if sample[feat_i] >= threshold:
            branch = tree.left_branch
        else:
            branch = tree.right_branch

        return self.predict_sample(sample, branch)
    
    def _impurity_calculation(self, left_y, right_y):
        raise NotImplementedError

    def _leaf_value_calculation(self, y):
        raise NotADirectoryError



class RegressionTree(DecisionTree):
    def _impurity_calculation(self, y, y1, y2):
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        # Calculate the variance reduction
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

        return sum(variance_reduction)

    def _leaf_value_calculation(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]




class ClassificationTree(DecisionTree):
    def _impurity_calculation(self, y, left_y, right_y):
        entropy = cal_entropy(y)
        p = len(left_y)/len(y)
        info_gain = entropy - p*cal_entropy(left_y) - (1-p)*cal_entropy(right_y)
        return info_gain

    def _leaf_value_calculation(self, y):
        most_common = None
        most_count = 0
        for label in np.unique(y):
            count = len(y[y==label])
            if count > most_count:
                most_common = label
                most_count = count
        return most_common




