import os

import PIL.Image as Image
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from thesis.data_prossesing.DataloaderSubsetDataset import DataloaderSubsetDataset



def data_reader_train(dir):
    image_dataset = datasets.ImageFolder(dir)

    data_aug_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # converts all images to (0,1]
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
    ])
    data_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        # converts all images to (0,1]
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
    ])

    train_size = int(0.9 * len(image_dataset))
    val_size = len(image_dataset) - train_size

    subset_train, subset_val = random_split(image_dataset, [train_size, val_size])

    train_dataset = DataloaderSubsetDataset(subset_train, transform=data_aug_transform)
    valid_dataset = DataloaderSubsetDataset(subset_val, transform=data_transform)

    # sampler = random_Sampler(image_dataset)

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=32,
                                                       shuffle=True,
                                                       # sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=False)

    valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=32,
                                                       shuffle=True,
                                                       # sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=False)

    dataloaders = {'train': train_dataset_loader,
                   'val': valid_dataset_loader}

    return dataloaders


def data_reader_test(dir):
    data_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        # converts all images to (0,1]
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
    ])

    image_dataset = datasets.ImageFolder(dir, transform=data_transform)

    test_dataloader = torch.utils.data.DataLoader(image_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  drop_last=True,
                                                  pin_memory=False)

    return test_dataloader


def data_reader(dir):
    image_dataset = datasets.ImageFolder(dir)

    data_aug_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # converts all images to (0,1]
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
    ])
    data_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        # converts all images to (0,1]
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
    ])

    train_size = int(0.6 * len(image_dataset))
    # val_size = (len(image_dataset) - train_size)/2
    val_size = int(0.15 * len(image_dataset))
    test_size = len(image_dataset) - train_size - val_size

    print("Train size: ", train_size)
    print("Val size: ", val_size)
    print("Test size: ", test_size)
    print("All images: ", train_size + val_size + test_size)

    subset_train, subset_val, subset_test = random_split(image_dataset, [train_size, val_size, test_size])

    train_dataset = DataloaderSubsetDataset(subset_train, transform=data_transform)
    valid_dataset = DataloaderSubsetDataset(subset_val, transform=data_transform)
    test_dataset = DataloaderSubsetDataset(subset_test, transform=data_transform)

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=32,
                                                       shuffle=True,
                                                       # sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=False)

    valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=32,
                                                       shuffle=True,
                                                       # sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=False)

    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      # sampler=sampler,
                                                      drop_last=True,
                                                      pin_memory=False)

    dataloaders = {'train': train_dataset_loader,
                   'val': valid_dataset_loader,
                   'test': test_dataset_loader}

    return dataloaders


def data_reader_chestxray(dir):
    # image_dataset = datasets.ImageFolder(dir)

    data_aug_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # converts all images to (0,1]
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
    ])
    data_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        # converts all images to (0,1]
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(dir + '/train', transform=data_transform)
    valid_dataset = datasets.ImageFolder(dir + '/val', transform=data_transform)
    test_dataset = datasets.ImageFolder(dir + '/test', transform=data_transform)

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=32,
                                                       shuffle=True,
                                                       # sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=False)

    valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=4,
                                                       shuffle=True,
                                                       # sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=False)

    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      # sampler=sampler,
                                                      drop_last=True,
                                                      pin_memory=False)

    dataloaders = {'train': train_dataset_loader,
                   'val': valid_dataset_loader,
                   'test': test_dataset_loader}

    return dataloaders


def print_image(image):
    image = np.squeeze(image)
    image = torch.transpose(image, 0, 2)
    plt.imshow(image)
    plt.show()


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
