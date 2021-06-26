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

with open('../config.yaml', 'r') as f:
    params = yaml.safe_load(f)

# set device
device = get_default_device()
print("Device:", device)

model = ResNet50()
model.freeze()
model.to(device)


# load dataset
data_dir = '../data/i2a2-brasil-pneumonia-classification'
dataloaders = data_reader(data_dir)

criterion = nn.BCELoss()
num_epochs = 10

epochs = 20
lr = 0.0001
grad_clip = None

# weighted loss for data class imbalance
weight = torch.FloatTensor([3876 / (1342 + 3876), 1342 / (1342 + 3876)]).to(device)

history, optimizer, best_loss = fit(epochs,
                                    lr,
                                    model,
                                    dataloaders['train'],
                                    dataloaders['val'],
                                    weight)
