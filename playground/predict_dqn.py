import os

import torch
import torch.nn as nn
import torch.optim as optim

from thesis.playground.dqn import DQNAgent

from thesis.data_prossesing.data_pytorch import data_reader, get_default_device
from thesis.playground.buffer import ReplayMemory

# set device
device = get_default_device()
print("Device:", device)

# load dataset
# data_dir = '../data/train_i2a2_complete/data'
data_dir = os.path.join('..', 'data', 'trainFolder', 'data')
dataloaders = data_reader(data_dir)
data = next(iter(dataloaders['test']))
image = data[0].to("cuda")
agent = DQNAgent(batch_size=4)

for e in range(20):
    action = agent.select_action(image)
    print(action)

# buffer = ReplayMemory(1000)
#
# buffer.push(data[0],2,1,0, False)
#
# state, action_num, reward, next_state, done = buffer.sample(1)
#
# state = state.pop(0).to(device)
# action_num = action_num.pop(0)
# reward = reward.pop(0)
# next_state = next_state.pop(0)
# done = done.pop(0)
#
# print(state)
# print(action_num)
# print(reward)
# print(next_state)
# print(done)
#
# print(agent.target_net.test(image))
#
