import time
import torch
from thesis.model.ResNet50_classifier import ResNet50
from thesis.data_prossesing.data_pytorch import data_reader_test
from thesis.data_prossesing.classifier_utils import get_default_device
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import pandas as pd

import yaml
import copy

CUDA_LAUNCH_BLOCKING=1

with open('../config.yaml', 'r') as f:
    params = yaml.safe_load(f)

# set device
device = get_default_device()
print("Device:", device)

model = ResNet50()
model.to(device)

# load dataset
data_dir = '../data/trainFolder/data'
dataloader = data_reader_test(data_dir)


weight_dir = os.path.join('..','weights','ResNet50_weights_pytorch.hdf5')
torch.load(weight_dir)


prediction = model.test_model(dataloader,device)



