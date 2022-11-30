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


class ResNet50Rl(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.network = torchvision.models.resnet50(pretrained=True)
            print("ResNet status: PRETRAINED")
        else:
            self.network = torchvision.models.resnet50(pretrained=False)
            print("ResNet status: NOT PRETRAINED")

        for param in self.network.parameters():
            param.require_grad = False

        num_in_features = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(num_in_features, 3),
            nn.Softmax(dim=1))

    def forward(self, x):
        return self.network(x)
