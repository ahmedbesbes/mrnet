import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as data
from torchvision import transforms
from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine


INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73


class MRDataset(data.Dataset):
    def __init__(self, data_path, task, plane, train=True, weights=None, transform=None):
        super().__init__()
        self.task = task
        self.plane = plane
        self.data_path = data_path
        if train:
            self.path = self.data_path + 'train/{0}/'.format(plane)
            self.path_files = [self.path + p for p in os.listdir(self.path)]
            self.labels = pd.read_csv(self.data_path +
                                      'train-{0}.csv'.format(task), header=None)[1]
        else:
            self.path = self.data_path + 'valid/{0}/'.format(plane)
            self.path_files = [self.path + p for p in os.listdir(self.path)]
            self.labels = pd.read_csv(
                self.data_path + 'valid-{0}.csv'.format(task), header=None)[1]

        self.transform = transform
        if weights is None:
            pos = np.sum(self.labels)
            neg = len(self.labels) - pos
            self.weights = [1, neg / pos]
        else:
            self.weights = weights

    def __len__(self):
        return len(self.path_files)

    def __getitem__(self, index):
        array = np.load(self.path_files[index])
        label = self.labels[index]
        label = torch.FloatTensor([label])

        # crop middle
        pad = int((array.shape[2] - INPUT_DIM)/2)
        array = array[:, pad:-pad, pad:-pad]

        # standardize
        array = (array - np.min(array)) / \
            (np.max(array) - np.min(array)) * MAX_PIXEL_VAL

        # normalize
        array = (array - MEAN) / STDDEV

        if self.transform:
            array = self.transform(array)
        else:
            array = np.stack((array,)*3, axis=1)

        if label.item() == 1:
            weight = np.array([self.weights[1]])
            weight = torch.FloatTensor(weight)
        else:
            weight = np.array([self.weights[0]])
            weight = torch.FloatTensor(weight)

        return array, label, weight
