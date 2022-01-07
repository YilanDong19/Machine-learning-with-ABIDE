import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
import os
import scipy.io as scio

def get_index3(lst=None, item=''):
    return [i for i in range(len(lst)) if lst[i] == item]
def flatten_one(img):
    shape = img.shape
    length = shape[0] * (shape[1]-1) / 2
    one_line = np.zeros((1, int(length)))
    position = 0
    for i in range(shape[0]): # column
        for j in range(i+1, shape[1]): # row
            one_line[0,position] = img[j,i]
            position = position+1
    return one_line

def feature_selection(matrix, labels, train_ind, fnum):

    estimator = RidgeClassifier()
    selector = RFE(estimator, fnum, step=100, verbose=1)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    return selector

def get_subject_score(subject_list,l, path):

    name = l + '.mat'
    file = scio.loadmat(os.path.join(path, name))
    file = file[l]
    label_dict = {}
    for i in subject_list:
        value = file[int(i)]
        if l == 'genders':
            label_dict[i] = int(value)
        elif l == 'ages':
            label_dict[i] = float(value)
        elif l == 'FIQS':
            label_dict[i] = float(value)
        elif l == 'NUM':
            label_dict[i] = int(value)
        elif l == 'PEC':
            label_dict[i] = float(value)
        elif l == 'RAT':
            label_dict[i] = int(value)
        elif l == 'sites':
            label_dict[i] = value.replace(' ', '')
        else:
            label_dict[i] = value
    return label_dict


def create_affinity_graph_from_scores(scores, subject_list, path):
    """
        scores       : list of phenotypic information to be used to construct the affinity graph
        subject_list : list of subject IDs
    return:
        graph        : adjacency matrix of the population graph (num_subjects x num_subjects)
    """

    num_nodes = len(subject_list)
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = get_subject_score(subject_list, l, path)

        # quantitative phenotypic scores
        if l in ['ages']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[subject_list[k]]) - float(label_dict[subject_list[j]]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass
        elif l in ['FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[subject_list[k]]) - float(label_dict[subject_list[j]]))
                        if val < 10:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[subject_list[k]] == label_dict[subject_list[j]]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph


