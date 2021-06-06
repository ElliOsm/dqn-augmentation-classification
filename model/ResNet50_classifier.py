import pandas as pd
import numpy as np
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision


class ResNet50(nn.Module):


    def __init__(self):
        super().__init__()
        self.network = torchvision.models.resnet50(pretrained=True)

        num_in_features = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(num_in_features, 16),
            nn.ReLU(),
            nn.Linear(16,1))

    # def forward(self, x):
    #     return x
    def forward(self, x):
        return torch.sigmoid(self.network(x))


    def freeze(self):
        for param in self.network.parameters():
            param.require_grad = False

        for param in self.network.fc.parameters():
            param.require_grad = True

    def train_model(model, dataloaders, device, optimizer, criterion, num_epochs):
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

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

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

        return model
