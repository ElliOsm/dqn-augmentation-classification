import pandas as pd
import torch
import torch.nn.functional as F
import PIL.Image as Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
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


#
# def data_reader(dir):
#     image_dataset = datasets.ImageFolder(dir)
#
#     img_csv = pd.read_csv(dir + '/train.csv')
#
#
#     data_aug_transform = transforms.Compose([
#         transforms.Resize([int(params['img_height']), int(params['img_width'])]),
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         # converts all images to (0,1]
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ])
#     data_transform = transforms.Compose([
#         transforms.Resize([int(params['img_height']), int(params['img_width'])]),
#         # converts all images to (0,1]
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ])
#
#     train_size = int(0.7 * len(image_dataset))
#     val_size = len(image_dataset) - train_size
#     # print('Size:')
#     # print('train length: ', train_size)
#     # print('val length: ', val_size)
#     subset_train, subset_val = random_split(image_dataset, [train_size, val_size])
#
#     # print('Subset:')
#     # print('train length: ', len(subset_train))
#     # print('val length: ', len(subset_val))
#
#     train_dataset = DataloaderSubsetDataset(subset_train, dir, transform=data_aug_transform)
#     valid_dataset = DataloaderSubsetDataset(subset_val, dir, transform=data_transform)
#
#     # print('Dataset:')
#     # print('train length: ', train_dataset.__len__())
#     # print('val length: ', valid_dataset.__len__())
#
#
#     train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
#                                                        batch_size=params['batch_size'],
#                                                        shuffle=True,
#                                                        drop_last=True)
#
#     valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
#                                                        batch_size=params['batch_size'],
#                                                        shuffle=True,
#                                                        drop_last=True)
#     # print('Dataloaders:')
#     # print('train length: ', train_dataset.__len__())
#     # print('val length: ', valid_dataset.__len__())
#
#     dataloaders = {'train': train_dataset_loader,
#                    'val': valid_dataset_loader}
#
#     return dataloaders
#

def data_reader(dir):
    image_dataset = datasets.ImageFolder(dir)


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

    train_size = int(0.9 * len(image_dataset))
    val_size = len(image_dataset) - train_size

    subset_train, subset_val = random_split(image_dataset, [train_size, val_size])


    train_dataset = DataloaderSubsetDataset(subset_train, transform=data_aug_transform)
    valid_dataset = DataloaderSubsetDataset(subset_val, transform=data_transform)

    #sampler = random_Sampler(image_dataset)

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=params['batch_size'],
                                                       shuffle=True,
                                                       #sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=False)

    valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=params['batch_size'],
                                                       shuffle=True,
                                                       #sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=False)

    dataloaders = {'train': train_dataset_loader,
                   'val': valid_dataset_loader}

    return dataloaders


def data_reader_test(dir):

    data_transform = transforms.Compose([
        transforms.Resize([int(params['img_height']), int(params['img_width'])]),
        # converts all images to (0,1]
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image_dataset = datasets.ImageFolder(dir,transform=data_transform)


    test_dataloader = torch.utils.data.DataLoader(image_dataset,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       drop_last=True,
                                                       pin_memory=False)


    return test_dataloader


def load_image(img_path):
    image = Image.open(img_path)
    prediction_transform =  transforms.Compose([
        transforms.Resize([int(params['img_height']), int(params['img_width'])]),
        # converts all images to (0,1]
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


    image = prediction_transform(image).unsqueeze(0)
    return image

def random_Sampler(image_dataset):
    class_sample_count = np.array(
        [len(np.where(image_dataset.targets == t)[0]) for t in np.unique(image_dataset.targets)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in image_dataset.targets])
    print(class_sample_count)
    print(len(class_sample_count))
    print(weight)
    print(samples_weight)
    print(len(samples_weight))

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    print(samples_weight)
    print(len(samples_weight))
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler

    # class_sample_count = [1108, 2991]
    # weights = 1 / torch.Tensor(class_sample_count)
    #
    # print(1108+ 2991)
    # print(len(weights))
    #
    #
    # weights = weights[target_list]
    #
    # #weights = weights.double()
    # sampler = WeightedRandomSampler(weights=weights,
    #                                 num_samples=4099,
    #                                 replacement=True)
    # return sampler



def data_reader(dir):
    image_dataset = datasets.ImageFolder(dir)


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

    train_size = int(0.6 * len(image_dataset))
   # val_size = (len(image_dataset) - train_size)/2
    val_size = int(0.15 * len(image_dataset))
    test_size = len(image_dataset) - train_size - val_size

    print(train_size, val_size, test_size)
    print(train_size + val_size + test_size)


    subset_train, subset_val, subset_test = random_split(image_dataset, [train_size, val_size,test_size])


    train_dataset = DataloaderSubsetDataset(subset_train, transform=data_aug_transform)
    valid_dataset = DataloaderSubsetDataset(subset_val, transform=data_aug_transform)
    test_dataset = DataloaderSubsetDataset(subset_test, transform=data_aug_transform)

    #sampler = random_Sampler(image_dataset)

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=params['batch_size'],
                                                       shuffle=True,
                                                       #sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=False)

    valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=params['batch_size'],
                                                       shuffle=True,
                                                       #sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=False)

    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=1,
                                                       shuffle=True,
                                                       #sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=False)



    dataloaders = {'train': train_dataset_loader,
                   'val': valid_dataset_loader,
                   'test':test_dataset_loader}

    return dataloaders


def data_reader_chestxray(dir):
    #image_dataset = datasets.ImageFolder(dir)


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



    train_dataset = datasets.ImageFolder(dir + '/train',transform = data_aug_transform)
    valid_dataset = datasets.ImageFolder(dir + '/val',transform = data_transform)
    test_dataset = datasets.ImageFolder(dir + '/test',transform = data_transform)


    #sampler = random_Sampler(image_dataset)

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=params['batch_size'],
                                                       shuffle=True,
                                                       #sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=False)

    valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
                                                       batch_size=4,
                                                       shuffle=True,
                                                       #sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=False)

    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=1,
                                                       shuffle=True,
                                                       #sampler=sampler,
                                                       drop_last=True,
                                                       pin_memory=False)



    dataloaders = {'train': train_dataset_loader,
                   'val': valid_dataset_loader,
                   'test': test_dataset_loader}

    return dataloaders