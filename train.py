import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import MRDataset
from model import MRNet
 

def train(model, trainloader, epoch, criterion, optimizer, num_epochs):
    model.train()
    train_losses = []
    for i, (image, label) in enumerate(trainloader):
        prediction = model(image)
        loss = criterion(label, prediction)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        train_losses.append(loss)
        print('[Epoch: {0} / {1} | Batch: {2} / {3} ]| train loss {4}'.format(
            epoch,
            num_epochs,
            i,
            len(trainloader),
            np.mean(train_losses)
        ))


def run(args):
    mrnet = MRNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(mrnet.parameters())

    trainset = MRDataset(args.task, args.plane, transform=bool(args.augment))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=8)

    for epoch in tqdm(range(args.epochs)):
        train(mrnet, trainloader, epoch, criterion, optimizer, args.epochs)

    print('finshed training')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['abnormal', 'acl', 'meniscus'])
    parser.add_argument('-p', '--plane', type=str, required=True,
                        choices=['saggital', 'coronal', 'axial'])
    parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
