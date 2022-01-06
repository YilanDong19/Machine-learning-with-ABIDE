

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
import os
import scipy.io as scio

def feature_selection(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)  特征矩阵
        labels       : ground truth labels (num_subjects x 1) 真标签
        train_ind    : indices of the training samples  训练样本的指标
        fnum         : size of the feature vector after feature selection 特征选择后特征向量的大小。

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum) 地为特征矩阵
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator, fnum, step=100, verbose=1)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    # x_data = selector.transform(matrix)

    return selector