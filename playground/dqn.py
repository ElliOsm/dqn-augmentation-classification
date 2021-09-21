import random

import math
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from thesis.playground.buffer import ReplayMemory
from thesis.playground.dqn_architectuer import DQN


class DQNAgent():

    def __init__(self, batch_size):
        self.actions = {
            0: self.action1,
            1: self.action2,
            2: self.action3
        }
        self.action_space = len(self.actions)

        self.learning_rate = 0.4
        self.gamma = 0.3
        self.episodes = 30
        self.exploration = 1
        self.exploration_threshold = 0.1

        self.q_eval_net = DQN().to("cuda")
        self.target_net = DQN().to("cuda")
        self.target_net.load_state_dict(self.q_eval_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.buffer = ReplayMemory(10000)

        self.batch_size = batch_size
        self.steps_done = 0

    # adjust_contrast
    def action1(self, image):
        image = np.squeeze(image)

        transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        image = transform(image)
        image = transforms.functional.adjust_contrast(image, contrast_factor=3)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image = transform(image)
        image = torch.unsqueeze(image, 0)

        return image

    # equalize image
    def action2(self, image):
        image = np.squeeze(image)

        transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        image = transform(image)
        image = transforms.functional.adjust_brightness(img=image, brightness_factor=2)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image = transform(image)
        image = torch.unsqueeze(image, 0)

        return image

    # sharpness
    def action3(self, image):
        image = np.squeeze(image)

        transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        image = transform(image)
        image = transforms.functional.adjust_sharpness(img=image, sharpness_factor=10)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        return image

    def select_action(self, image):
        if self.steps_done < 10:
            epsilon = self.exploration
        else:
            epsilon = self.exploration_threshold
        self.steps_done += 1
        # https://stackoverflow.com/questions/33359740/random-number-between-0-and-1-in-python
        if np.random.uniform(0, 1) < epsilon:
            action_num = random.randint(0, 2)
        else:
            action_num = self.q_eval_net(image)
            _, action_num = torch.max(action_num, dim=1)
            action_num = action_num.item()

        return action_num

    def apply_action(self, action_num, image):
        action = self.actions[action_num]
        image = action(image)
        return image

    def get_state(self, reward):
        if reward > 0:
            return 0
        else:
            return 1

    # def get_next_state(self, state):
    #     if state == 0:
    #         return 1
    #     elif state == 1:
    #         return - 1
    #     else:
    #         return 0

    # def is_done(self, state):
    #     if state == -1:
    #         state = 0
    #         return True, state
    #     else:
    #         return False, state

    def get_reward(self, m_before, m_after):
        difference = (m_after - m_before).float()
        if difference > 0:
            return 1
        elif difference == 0:
            return 0
        else:
            return -1

    def training(self):
        if len(self.buffer) < self.batch_size:
            return

        state, action_num, reward, next_state, done = self.buffer.sample(self.batch_size)

        state = state.pop(0).to(self.device)
        action_num = action_num.pop(0)
        reward = reward.pop(0)
        next_state = next_state.pop(0)
        # done = done.pop(0)

        q_eval = self.q_eval_net.test(state)
        q_next = self.target_net(next_state).detach()
        q_target = reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, dataloaders):
        if len(self.buffer) < self.batch_size:
            return

    def get_features(self, features):
        return features.std()

    def predict(self, model, image):
        features = model.extract_features(image)
        m_before = self.get_features(features)
        state = 0
        for e in range(self.episodes):
            action_num = self.select_action(state)
            image_after = self.apply_action(action_num, image)
            features_after = model.extract_features(image_after.to("cuda"))
            m_after = self.get_features(features_after)
            reward = self.get_reward(m_before, m_after)
            state_num = self.get_state(reward)
            # if state_num == 0:
            state = image
            next_state = image_after
            # else:
            #     state = image_after
            #     next_state = image
            # done, next_state = self.is_done(next_state)
            self.buffer.push(state, action_num, reward, next_state)
            self.training()
