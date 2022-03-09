import random
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


class DQN(nn.Module):
    # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def __init__(self, h=224, w=224, outputs=3):
        super(DQN, self).__init__()
        # same as resnet
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to("cuda")
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # https://stackoverflow.com/questions/62621317/why-return-self-headx-viewx-size0-1-in-the-nn-module-for-pytorch-reinfor
        return self.head(x.view(x.size(0), -1))

    def test(self, input):
        outputs = self(input)
        probs = nn.Softmax(dim=1)
        outputs = probs(outputs)
        _, preds = torch.max(outputs, 1)
        return preds