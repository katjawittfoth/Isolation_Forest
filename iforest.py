import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from lolviz import *
from multiprocessing import Pool

# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf


class CFactor:
    @classmethod
    def compute(cls, n_of_elements):
        n = n_of_elements
        if n > 2:
            return 2 * (np.log(n - 1) + np.euler_gamma) - (2 * (n - 1) / n)
        elif n == 2:
            return 1
        else:
            return 0


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = []
        self.height_limit = np.ceil(np.log2(self.sample_size))

    def make_tree(self, i):
        X_sample = self.X[np.random.choice(self.X.shape[0], self.sample_size, replace=False)]
        return IsolationTree(X_sample, self.height_limit, self.good_features, self.improved)

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert XFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame): X = X.values
        self.X = X
        self.improved = improved

        if self.improved:
            median_c = np.median(self.X, axis=0)
            mean_c = np.mean(self.X, axis=0)
            std_c = np.std(self.X, axis=0)
            result = 100 * (abs(median_c - mean_c) / std_c)
            thresh = np.mean(result, axis=0)
            self.good_features = np.where(result > thresh)[0]

            with Pool(5) as p:
                self.trees = p.map(self.make_tree, range(self.n_trees))

        else:
                X_sample = X[np.random.choice(X.shape[0], self.sample_size, replace=False)]
                self.trees.append(IsolationTree(X_sample, self.height_limit))


        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        if isinstance(X, pd.DataFrame): X = X.values

        length = []

        for x_i in X:
            x_len = []
            for tree in self.trees:
                l = tree.root.path_length(x_i)
                x_len.append(l)
            avg_len = np.array(x_len).mean()
            length.append([avg_len])

        return np.array(length)

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        avg_pathlen = self.path_length(X)
        c = CFactor.compute(self.sample_size)
        anom_score = np.exp2(-avg_pathlen/c)

        return anom_score

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return (scores>=threshold).astype(int)

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        return predict_from_anomaly_score(self, anomaly_score(self, X), threshold)


class Node:
    def __init__(self, left=None, right=None, split_attr=None, split_point=None, c_factor=None):
        self.left = left
        self.right = right
        self.split_attr = split_attr
        self.split_point = split_point
        self.c_factor = c_factor

    def path_length(self, x, current_height=0):
        if self.left == None and self.right == None:
            return current_height + self.c_factor

        if x[self.split_attr] < self.split_point:
            return self.left.path_length(x, current_height + 1)
        else:
            return self.right.path_length(x, current_height + 1)


class IsolationTree:
    def __init__(self, X:np.ndarray, height_limit, good_features=[], improved=False):
        self.height_limit = height_limit
        self.n_nodes = 0
        self.split_attr = None
        self.split_point = None
        self.good_features = good_features
        self.improved = improved
        self.root = self.fit(X, 0)

    def fit(self, X:np.ndarray, current_height):
        if ((current_height >= self.height_limit) or (len(X) <= 1)):
            c_factor = CFactor.compute(X.shape[0])
            return Node(None, None, -1, None, c_factor)

        self.n_nodes += 1
        node = Node()

        if self.improved:
            tooBalanced = True

            while tooBalanced:
                q = np.random.choice(self.good_features, replace=False)

                X_column = X[:, q]
                minv = X_column.min()
                maxv = X_column.max()

                if minv == maxv:
                    c_factor = CFactor.compute(X.shape[0])
                    return Node(None, None, -1, None, c_factor)

                p = float(np.random.uniform(minv, maxv))

                X_l = X[X_column < p, :]
                X_r = X[X_column >= p, :]

                node.split_point = p
                node.split_attr = q

                total = X.shape[0]
                ls = X_l.shape[0]
                rs = X_r.shape[0]

                diff = float(abs(ls - rs)/total)

                if diff >= 0.25 or total == 2:
                    tooBalanced = False

        else:
            node.split_attr = np.random.randint(0, X.shape[1])

            X_column = X[:, node.split_attr]
            minv = X_column.min()
            maxv = X_column.max()

            if minv == maxv:
                c_factor = CFactor.compute(X.shape[0])
                return Node(None, None, -1, None, c_factor)

            node.split_point = float(np.random.uniform(minv, maxv))

            X_l = X[X_column < node.split_point, :]
            X_r = X[X_column >= node.split_point, :]

        node.left = self.fit(X_l, current_height + 1)
        node.right = self.fit(X_r, current_height + 1)

        return node


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1.0
    tpr = 0.0
    fpr = 0.0

    while tpr < desired_TPR:
        threshold -= 0.01
        y_pred = (scores >= threshold).astype(int)
        confusion = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = confusion.flat
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
    return threshold, fpr
