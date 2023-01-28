import os

import torch
import torch.nn as nn
import torch.optim as optim

from thesis.data_prossesing.data_pytorch import data_reader
from thesis.data_prossesing.data_pytorch import get_default_device
from thesis.model.ResNet50_classifier import ResNet50

CUDA_LAUNCH_BLOCKING = 1

# set device
device = get_default_device()
print("Device:", device)

# create Model
model = ResNet50(pretrained=False)
model.to(device)

# load dataset
print("Dataset: TRAIN I2A2")
data_dir = os.path.join('..','data','train_i2a2_complete' ,'data')
dataloaders = data_reader(data_dir)

# https://datascience.stackexchange.com/questions/48369/what-loss-function-to-use-for-imbalanced-classes-using-pytorch
# create class weights
class_count = [1108, 2991]
w0 = (class_count[1]) / (class_count[0])
w1 = (class_count[1]) / (class_count[1])

class_weights = torch.FloatTensor([w0, w1]).to(device)
print("class_weights: ", class_weights)

# #https://stats.stackexchange.com/questions/260505/should-i-use-a-categorical-cross-entropy-or-binary-cross-entropy-loss-for-binary
# loss_func = nn.CrossEntropyLoss(weight=class_weights)
loss_func = nn.BCELoss(weight=class_weights)
optimiser = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 10

model = model.train_model(dataloaders, device, optimiser, loss_func, num_epochs)

weight_dir = os.path.join('..', 'weights', 'test_i2a2-brasil_not_pretrained.hdf5')
torch.save(model.state_dict(), weight_dir)
print("weights saved at: ", weight_dir)
