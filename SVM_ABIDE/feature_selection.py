import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
import os
import scipy.io as scio

def feature_selection(matrix, labels, train_ind, fnum):
    
    estimator = RidgeClassifier()
    selector = RFE(estimator, fnum, step=100, verbose=1)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())

    return selector
