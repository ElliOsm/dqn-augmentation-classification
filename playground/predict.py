import os

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from thesis.playground.dqn import DQNAgent

from thesis.data_prossesing.data_pytorch import data_reader, get_default_device

# set device
device = get_default_device()
print("Device:", device)


# load dataset
#data_dir = '../data/train_i2a2_complete/data'
data_dir = os.path.join('..','data','trainFolder' ,'data')
dataloaders = data_reader(data_dir)
data = next(iter(dataloaders['test']))
image = data[0].to("cuda")
agent = DQNAgent(batch_size=4)

for e in range(50):
    action = agent.select_action(image)
    print("out")
    print(action)
