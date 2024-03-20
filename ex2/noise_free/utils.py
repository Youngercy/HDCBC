#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import numpy as np

from scipy.optimize import linear_sum_assignment as linear_assignment



def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed.
    (Taken from https://github.com/XifengGuo/IDEC-toy/blob/master/DEC.py)
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    #print(y_pred.size)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    #print(w)
    ind = linear_assignment(w.max() - w)

    tuple_list = [(ind[0][i], ind[1][i]) for i in range(len(ind[0]))]

    correct_clusters = sum([w[i, j] for i, j in tuple_list])
    accuracy = correct_clusters * 1.0 / y_pred.size
    return accuracy
