import pandas as pd
import torch
import torch.nn.functional as F
import PIL.Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
from thesis.data_prossesing.DataloaderSubsetDataset import DataloaderSubsetDataset
import torchdata as td
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import yaml

with open('../config.yaml', 'r') as f:
    params = yaml.safe_load(f)


def one_hot_encodoing(x):
    return F.one_hot(x, params['num_class'])


def data_reader(dir):
    image_dataset = datasets.ImageFolder(dir)

    img_csv = pd.read_csv(dir + '/train.csv')


    data_aug_transform = transforms.Compose([
        transforms.Resize([int(params['img_height']), int(params['img_width'])]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # converts all images to (0,1]
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    data_transform = transforms.Compose([
        transforms.Resize([int(params['img_height']), int(params['img_width'])]),
        # converts all images to (0,1]
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_size = int(0.7 * len(image_dataset))
    val_size = len(image_dataset) - train_size
    # print('Size:')
    # print('train length: ', train_size)
    # print('val length: ', val_size)
    subset_train, subset_val = random_split(image_dataset, [train_size, val_size])

    # print('Subset:')
    # print('train length: ', len(subset_train))
    # print('val length: ', len(subset_val))

    train_dataset = DataloaderSubsetDataset(subset_train, dir, transform=data_aug_transform)
    valid_dataset = DataloaderSubsetDataset(subset_val, dir, transform=data_transform)

    # print('Dataset:')
    # print('train length: ', train_dataset.__len__())
    # print('val length: ', valid_dataset.__len__())


    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=params['batch_size'],
                                                       shuffle=True,
                                                       drop_last=True)

    valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=params['batch_size'],
                                                       shuffle=True,
                                                       drop_last=True)
    # print('Dataloaders:')
    # print('train length: ', train_dataset.__len__())
    # print('val length: ', valid_dataset.__len__())

    dataloaders = {'train': train_dataset_loader,
                   'val': valid_dataset_loader}

    return dataloaders

