import time
import torch
from thesis.model.ResNet50_classifier import ResNet50
from thesis.data_prossesing.data_pytorch import data_reader,get_default_device
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

import copy

CUDA_LAUNCH_BLOCKING = 1

# set device
device = get_default_device()
print("Device:", device)

model = ResNet50()
model.freeze()
model.to(device)

# load dataset
data_dir = os.path.join('..','data','train_i2a2_complete' ,'data')
dataloaders = data_reader(data_dir)

weight_dir = os.path.join('..', 'weights', 'test_i2a2-brasil.hdf5')
model.load_state_dict(torch.load(weight_dir))
print("weights loaded successfully from: ", weight_dir)


prediction = model.test_model(dataloaders['test'], device)
