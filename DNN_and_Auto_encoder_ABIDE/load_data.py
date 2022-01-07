import os
import numpy as np
from torch.utils.data.dataset import Dataset
from os import listdir
import scipy.io as scio


class TrainDataset_DNN(Dataset):

    def __init__(self, root_dir, label_dir, selector):
        '''
          root_dir - string - path towards the folder containg the data
        '''
        self.root_dir = root_dir+'\\'
        self.label_dir = label_dir+'\\'
        self.filenames = listdir(self.root_dir)
        self.selector = selector

    def __len__(self):
        return len(self.filenames)

    def flatten_one(self, img):
        shape = img.shape
        length = shape[0] * (shape[1]-1) / 2
        one_line = np.zeros((1, int(length)))
        position = 0
        for i in range(shape[0]): # column
            for j in range(i+1, shape[1]): # row
                one_line[0,position] = img[j,i]
                position = position+1
        return one_line

    def __getitem__(self, idx):
        # Fetch file filename
        index = self.filenames[idx]
        img_name = str(index)
        img = scio.loadmat(os.path.join(self.root_dir, img_name))
        img = img['connectivity']
        img = self.flatten_one(img)

        label_name = 'ABIDE_label_871.mat'
        label = scio.loadmat(os.path.join(self.label_dir, label_name))
        a = label['label']
        index = index[:-4]
        label = a[:,int(index)]


        img = self.selector.transform(img)
        img = np.expand_dims(img, 0)

        return img, label, index

