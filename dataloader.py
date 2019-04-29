import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as data
from torchvision import transforms
from torchsample.transforms import RandomRotate, RandomTranslate


INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73
 

class MRDataset(data.Dataset):
    def __init__(self, data_path, task, plane, train=True, weights=None, augment=True):
        super().__init__()
        self.task = task
        self.plane = plane
        self.data_path = data_path
        self.augment = augment
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

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

        if weights is None:
            pos_weight = np.mean(self.labels.values)
            self.weights = [pos_weight, 1]
        else:
            self.weights = weights

    def __len__(self):
        return len(self.path_files)

    def transform(self, mri):
        s = mri.shape[0]
        degrees = np.random.randint(-25, 25)
        pixels = np.random.randint(-25, 25)
        proba_flip = np.random.random()
        processed_slides = []
        for i in range(s):
            slide = mri[i]
            slide = self.to_pil(slide)
            slide = TF.affine(slide, degrees, [pixels, pixels], 1, 0)
            if proba_flip > 0.5:
                slide = TF.hflip(slide)
            slide = self.to_tensor(slide)
            slide = slide.unsqueeze(0)
            processed_slides.append(slide)
        processed_slides = torch.cat(processed_slides, 0)
        # processed_slides = processed_slides.unsqueeze(0)
        return processed_slides

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

        if self.augment:
            array = self.transform(array)

        if label.item() == 1:
            weight = np.array([self.weights[1]])
            weight = torch.FloatTensor(weight)
        else:
            weight = np.array([self.weights[0]])
            weight = torch.FloatTensor(weight)

        return array, label, weight
