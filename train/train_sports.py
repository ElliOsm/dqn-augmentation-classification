import os

import torch
import torch.nn as nn
import torch.optim as optim

from thesis.data_prossesing.data_pytorch import data_reader
from thesis.data_prossesing.data_pytorch import get_default_device
from thesis.model.ResNet50_classifier_for_7_classes import ResNet50

CUDA_LAUNCH_BLOCKING = 1

# set device
device = get_default_device()
print("Device:", device)

model = ResNet50()
model.freeze()
model.to(device)

# load dataset
# data_dir = '../data/train_i2a2_complete/data'
print("Dataset: Sport Dataset")
data_dir = os.path.join('..', 'data', 'sports_complete', 'data')
dataloaders = data_reader(data_dir)

# https://datascience.stackexchange.com/questions/48369/what-loss-function-to-use-for-imbalanced-classes-using-pytorch
class_count = [1556, 1471, 578, 1445, 1394, 1188, 595]
w0 = (class_count[0]) / (class_count[0])
w1 = (class_count[0]) / (class_count[1])
w2 = (class_count[0]) / (class_count[2])
w3 = (class_count[0]) / (class_count[3])
w4 = (class_count[0]) / (class_count[4])
w5 = (class_count[0]) / (class_count[5])
w6 = (class_count[0]) / (class_count[6])

class_weights = torch.FloatTensor([w0, w1, w2, w3, w4, w5, w6]).to(device)
print("class_weights: ", class_weights)


# #https://stats.stackexchange.com/questions/260505/should-i-use-a-categorical-cross-entropy-or-binary-cross-entropy-loss-for-binary
loss_func = nn.CrossEntropyLoss(weight=class_weights)
optimiser = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 15

model = model.train_model(dataloaders, device, optimiser, loss_func, num_epochs)

weight_dir = os.path.join('..', 'weights', 'train_sports_dataset.hdf5')
torch.save(model.state_dict(), weight_dir)
print("weights saved at: ", weight_dir)
