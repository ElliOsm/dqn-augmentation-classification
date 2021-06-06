import time
import torch
from thesis.model.ResNet50_classifier import ResNet50
from thesis.data_prossesing.data_pytorch import data_reader
import numpy as np
import torch.nn as nn
import torch.optim as optim

import yaml
import copy

with open('../config.yaml', 'r') as f:
    params = yaml.safe_load(f)


#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = ResNet50()

#load dataset
data_dir = '../data/i2a2-brasil-pneumonia-classification'
dataloaders, data_size = data_reader(data_dir)

#model = model.train_model(device,dataloaders,data_size)

