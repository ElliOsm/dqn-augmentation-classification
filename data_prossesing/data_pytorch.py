import os

import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from thesis.data_prossesing.DataloaderSubsetDataset import DataloaderSubsetDataset


def data_reader(dir):
    image_dataset = datasets.ImageFolder(dir)

    data_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        # converts all images to (0,1]
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
    ])

    train_size = int(0.6 * len(image_dataset))
    # val_size = (len(image_dataset) - train_size)/2
    val_size = int(0.25 * len(image_dataset))
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
                                                       pin_memory=True)

    valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=1,
                                                       shuffle=True,
                                                       # sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=True)

    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      # sampler=sampler,
                                                      drop_last=True,
                                                      pin_memory=True)

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

