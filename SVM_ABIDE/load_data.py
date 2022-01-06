import os
import argparse
import numpy as np
import time
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
import torchvision.utils as vutils
from os.path import join
from os import listdir
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from collections import OrderedDict
import nibabel as nib
import matplotlib.pyplot as plt
import cv2 as cv
from os import path
import shutil
import scipy.stats
import scipy.ndimage
import random
import torch.nn.init as init
import torch.nn.functional as F
import sys
import math
from functools import reduce
import operator
from scipy.interpolate import interp1d
import nibabel as nib
import SimpleITK as sitk
import scipy.io as scio
import matplotlib


def obtain_onehot_site(all_sites, site):
    unique = np.unique(all_sites)
    unique = list(unique)
    one_hot_site = np.zeros((1, len(unique)))
    for i in range(len(unique)):
        if unique[i] == site:
            one_hot_site[0, i] = 1

    return one_hot_site

def flatten_one(length, img):
    '''
      In some situations, the dimension on z-axis of images are smaller than
      the dimension of z-axis of patches, this function will be used to pad
    '''

    one_line = np.zeros((1, int(length)))
    position = 0
    for i in range(img.shape[0]):  # column
        for j in range(i + 1, img.shape[1]):  # row
            one_line[0, position] = img[j, i]
            position = position + 1
    return one_line


def load_data_SVM(image_path, image_size, label_path, label_name, selector=None, new_number_features = None):


    filenames = listdir(image_path)
    number = len(filenames)
    length = image_size[0] * (image_size[1]-1) / 2
    group_images = np.zeros((number, new_number_features))
    group_labels = np.zeros((number, 1))

    labels = scio.loadmat(os.path.join(label_path, label_name))
    labels = labels['label']
    for index in range(number):
        image_name = str(filenames[index])
        image = scio.loadmat(os.path.join(image_path, image_name))
        img = image['connectivity']
        img = flatten_one(length, img)
        group_images[index,:] = selector.transform(img)

        label_position = image_name[:-4]
        group_labels[index,0] = labels[0, int(label_position)]

    return group_images, group_labels











