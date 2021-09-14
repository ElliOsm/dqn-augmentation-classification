import os

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from thesis.data_prossesing.data_pytorch import data_reader, get_default_device
from thesis.model.ResNet50_classifier import ResNet50

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

item = next(iter(dataloaders['test']))
print(item[0].shape)

# https://datascience.stackexchange.com/questions/48369/what-loss-function-to-use-for-imbalanced-classes-using-pytorch
class_count = [1108, 2991]
w0 = (class_count[1]) / (class_count[0])
w1 = (class_count[1]) / (class_count[1])

class_weights = torch.FloatTensor([w0, w1]).to(device)
print("class_weights: ", class_weights)

loss_func = nn.CrossEntropyLoss(weight=class_weights)
# loss_func = nn.BCELoss()
optimiser = optim.Adam(model.parameters(), lr=0.0001)
# optimiser = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
num_epochs = 10

model = model.train_model(dataloaders, device, optimiser, loss_func, num_epochs)

weight_dir = os.path.join('..', 'weights', 'test_train_i2a2-brasil.hdf5')
torch.save(model.state_dict(), weight_dir)

prediction = model.test_model(dataloaders['test'], device)
