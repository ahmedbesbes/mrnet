import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torchsample.transforms import RandomRotate, RandomTranslate


INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73


class MRDataset(data.Dataset):
    def __init__(self, data_path, task, plane, train=True, transform=True):
        super().__init__()
        self.task = task
        self.plane = plane
        self.transform = transform
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

        self.pos_weight = np.mean(self.labels)
        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(25),
            transforms.RandomAffine(0, translate=(0.11, 0.11)),
            transforms.RandomHorizontalFlip()
        ])

    def __len__(self):
        return len(self.path_files)

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
        label = torch.FloatTensor([label])

        if self.transform:
            array = self.data_transform(array)

        if label.item() == 1:
            weight = np.array([1 - self.pos_weight])
            weight = torch.FloatTensor(weight)
        else:
            weight = np.array([self.pos_weight])
            weight = torch.FloatTensor(weight)

        return array, label, weight 