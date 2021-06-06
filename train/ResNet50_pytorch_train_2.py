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


model = torchvision.models.resnet50(pretrained=True)

num_in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_in_features, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid())

for param in model.parameters():
    param.require_grad = False

for param in model.fc.parameters():
    param.require_grad = True

model.to(device)
# print("Model:", model)


# load dataset
data_dir = '../data/i2a2-brasil-pneumonia-classification'
dataloaders = data_reader(data_dir)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10

since = time.time()
best_acc = 0.0

for epoch in range(1, num_epochs + 1):
    print('Epoch {}/{}'.format(epoch, num_epochs))

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        optimizer.zero_grad()

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            labels = labels.to(torch.float32)
            loss = criterion(outputs, labels.unsqueeze(1))

            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects / len(dataloaders[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

