import torch.nn as nn

import torchvision


class ResNet50Rl(nn.Module):

    def __init__(self, pretrained=False):
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
            nn.Linear(num_in_features, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,3),
            nn.Softmax(dim=1))

    def forward(self, x):
        return self.network(x)
