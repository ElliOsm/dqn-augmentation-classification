import os

import torch

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
data_dir = os.path.join('..','data','trainFolder' ,'data')
dataloaders = data_reader(data_dir)

weight_dir = os.path.join('..','weights','train_chestxray_dataset.hdf5')
load_weight = model.load_state_dict(torch.load(weight_dir))


prediction = model.test_model(dataloaders['test'], device)
