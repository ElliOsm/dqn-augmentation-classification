# from thesis.model import drq
import time
import torch

from scipy.ndimage.interpolation import rotate

from thesis.model.qlearning import DQNAgent
from thesis.model.ResNet50_classifier import ResNet50
from thesis.data_prossesing.data_pytorch import data_reader,print_image,get_default_device
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import hydra

import yaml
import copy



CUDA_LAUNCH_BLOCKING = 1

with open('../config.yaml', 'r') as f:
    params = yaml.safe_load(f)

# set device
device = get_default_device()
print("Device:", device)

# load dataset
data_dir = os.path.join('..','data','train_i2a2_complete' ,'data')
dataloaders = data_reader(data_dir)

data = next(iter(dataloaders['test']))
data2 = next(iter(dataloaders['test']))

# resnet creation
model = ResNet50()
model.freeze()
model.to(device)

# weight load
weight_dir = os.path.join('..', 'weights', 'test_i2a2-brasil.hdf5')
load_weight = model.load_state_dict(torch.load(weight_dir))


#agent creation
agent = DQNAgent()
correct_after = 0
correct_before = 0
total = 0
all_after = 0

model.eval()
for inputs, labels in dataloaders['test']:
    inputs = inputs.to(device)
    labels = labels.to(device)
    predicted_label_before = model(inputs)
    _, preds = torch.max(predicted_label_before, dim=1)
    if (preds != labels):
        agent.find_best_action(inputs, model)
        action = agent.choose_final_action()
        image_after = agent.apply_action(action, inputs)
        
        predicted_label_after = model(inputs)
        _, preds_after = torch.max(predicted_label_after, dim=1)

        if preds_after == labels:
            correct_after = correct_after + 1
        all_after += labels.size(0)
    else:
        correct_before = correct_before + 1
    total += labels.size(0)

print("Correct without RL: ", correct_before,"/", total)
print("Correct with RL: ", correct_after + correct_before , "/", total)
print("Correct only RL: ", correct_after , "/", all_after)
print("All Correct: ", correct_before + correct_after , "/", total)