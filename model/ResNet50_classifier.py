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


class ResNet50(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.network = torchvision.models.resnet50(pretrained=True)
            print("ResNet status: PRETRAINED")
        else:
            self.network = torchvision.models.resnet50(pretrained=False)
            print("ResNet status: NOT PRETRAINED")

        num_in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(num_in_features, 2)
        self.threshold = 0.8

    def forward(self, x):
        return self.network(x)

    def freeze(self):
        for param in self.network.parameters():
            param.require_grad = False

        for param in self.network.fc.parameters():
            param.require_grad = True

    def train_model(self, dataloaders, device, optimizer, loss_func, num_epochs):
        torch.cuda.empty_cache()

        since = time.time()
        best_acc = 0.0

        best_model_wts = self.state_dict()

        for epoch in range(1, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)

            for phase in ['train', 'val']:
                since_epoch = time.time()
                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = self(inputs)

                    probs = nn.Softmax(dim=1)
                    outputs = probs(outputs)

                    _, preds = torch.max(outputs, 1)
                    # labels = labels.to(torch.float32)
                    loss = loss_func(outputs, labels)

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
                    best_model_wts = copy.deepcopy(self.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        self.load_state_dict(best_model_wts)
        return self

    def test_model(self, dataloader, device):
        counter = 0
        all = 0
        correct = 0
        total = 0
        with torch.no_grad():
            self.eval()
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # calculate outputs by running images through the network
                outputs = self(inputs)

                # https://stackoverflow.com/questions/59687382/testing-and-confidence-score-of-network-trained-with-nn-crossentropyloss
                probs = nn.Softmax(dim=1)
                outputs = probs(outputs)
                _, preds = torch.max(outputs, dim=1)
                # # print(preds)
                # # compare predictions to true label
                if preds == labels:
                    correct = correct + 1
                total = total + 1
        print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
            100. * correct / total, correct, total))
        # print("Confidense score",counter, "/", all)

    def test_image(self,inputs):
        with torch.no_grad():
            self.eval()
            # calculate outputs by running images through the network
            outputs = self(inputs)
            probs = nn.Softmax(dim=1)
            outputs = probs(outputs)
            _, preds = torch.max(outputs, dim=1)
            return preds


    # https://discuss.pytorch.org/t/how-can-l-use-the-pre-trained-resnet-to-extract-feautres-from-my-own-dataset/9008/2
    def extract_features(self, image):
        model = self.network
        modules = list(model.children())[:-1]
        resnet50 = nn.Sequential(*modules)
        for p in resnet50.parameters():
            p.requires_grad = False
        features = resnet50(image)
        return features

    def extract_propabilities(self, image):
        with torch.no_grad():
            self.eval()
            outputs = self(image)
            probs = nn.Softmax(dim=1)
            outputs = probs(outputs)
            return outputs

    def get_classification_result(self,image):
        outputs = self(image)
        _, output_label = torch.max(outputs, dim=1)
        return output_label