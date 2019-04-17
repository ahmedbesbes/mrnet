import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73


class Dataset(data.Dataset):
    def __init__(self, task, plane, train=True):
        super().__init__()
        self.task = task
        self.plane = plane
        if train:
            self.path = './data/train/{0}/'.format(plane)
            self.path_files = [self.path + p for p in os.listdir(self.path)]
            self.labels = pd.read_csv(
                './train/train-{0}.csv'.format(task), header=None)[1]
        else:
            self.path = './data/valid/{0}/'.format(plane)
            self.path_files = [self.path + p for p in os.listdir(self.path)]
            self.labels = pd.read_csv(
                './train/valid-{0}.csv'.format(task), header=None)[1]

    def __getitem__(self, index):
        array = np.load(self.path_files[index])
        label = self.labels[index]

        # crop middle
        pad = int((array.shape[2] - INPUT_DIM)/2)
        array = array[:, pad:-pad, pad:-pad]

        # standardize
        array = (array - np.min(array)) / \
            (np.max(array) - np.min(array)) * MAX_PIXEL_VAL

        # normalize
        array = (array - MEAN) / STDDEV

        # convert to RGB
        array = np.stack((array,) * 3, axis=1)

        array = torch.FloatTensor(array)
        label = torch.FloatTensor(label)

        return array, label