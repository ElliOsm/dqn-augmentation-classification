import pandas as pd
import numpy as np
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from thesis.data_prossesing.data_pytorch import load_image



class ResNet50(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = torchvision.models.resnet50(pretrained=True)

        num_in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(num_in_features, 2)


    def forward(self, x):
        return self.network(x)


    def freeze(self):
        for param in self.network.parameters():
            param.require_grad = False

        for param in self.network.fc.parameters():
            param.require_grad = True

    def train_model(model, dataloaders, device, optimizer, criterion, num_epochs):
        torch.cuda.empty_cache()

        since = time.time()
        best_acc = 0.0

        best_model_wts = model.state_dict()

        for epoch in range(1, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)

            for phase in ['train', 'val']:
                since_epoch = time.time()
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # print(labels)


                    outputs = model(inputs)
                    # print(outputs)
                    _, preds = torch.max(outputs, 1)
                    #labels = labels.to(torch.float32)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
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
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)
        return model

    # def test_model(model, dataloader, device, criterion):
    #     correct = 0.
    #     total = 0.
    #     for inputs, labels in dataloader:
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         output = model(inputs)
    #
    #         loss = criterion(output, labels)
    #
    #         _, preds = torch.max(output, 1)
    #         # compare predictions to true label
    #         if (preds == labels):
    #             correct = correct + 1
    #         total += labels.size(0)
    #         print(correct)
    #
    #     print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
    #         100. * correct / total, correct, total))


    def test_model(model, dataloader, device):
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # calculate outputs by running images through the network
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim=1)
                # compare predictions to true label
                if (preds == labels):
                    correct = correct + 1
                total += labels.size(0)
        print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
            100. * correct / total, correct, total))

