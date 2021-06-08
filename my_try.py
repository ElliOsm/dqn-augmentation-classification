import time
import torch
from thesis.model.ResNet50_classifier import ResNet50
from thesis.data_prossesing.data_pytorch import data_reader
from thesis.data_prossesing.classifier_utils import get_default_device

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

import yaml
import copy

CUDA_LAUNCH_BLOCKING=1

with open('../config.yaml', 'r') as f:
    params = yaml.safe_load(f)

# set device
device = get_default_device()
print("Device:", device)

model = ResNet50()
model.freeze()
model.to(device)


# load dataset
data_dir = '../data/testFolder/data'
dataloaders = data_reader(data_dir)



#criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(),lr=0.0001)
num_epochs = 20


model = model.train_model(dataloaders,device,optimiser,criterion,num_epochs)