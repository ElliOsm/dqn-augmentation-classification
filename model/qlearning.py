import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from thesis.model.ResNet50_classifier import ResNet50


class qlearning:

    def __init__(self):
        self.actions = {
            0: self.action1,
            1: self.action2,
            2: self.action3
        }
        self.qTable = np.zeros((2, 3))  # Q = np.zeros((state_size, action_size))
        self.state = [0, 1]
        self.learning_rate = 0.4
        self.discount_factor = 0.3
        self.episodes = 50

    #
    # # horizontal flip
    # def action3(self, image):
    #     unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #     unorm(image)
    #
    #     transform = transforms.Compose([
    #         transforms.ToPILImage()
    #     ])
    #     image = transform(image)
    #     image = transforms.functional.hflip(img=image)
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406],
    #                              [0.229, 0.224, 0.225])
    #     ])
    #
    #     return transform(image)

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

    def select_random_action(self):
        return random.randint(0, 2)

    def apply_action(self, action_num, image):
        action = self.actions[action_num]
        image = action(image)
        return image

    def get_reward(self, m_before, m_after):
        difference = (m_after - m_before).float()
        if difference > 0:
            return 1
        elif difference == 0:
            return 0
        else:
            return -1

    def get_state(self, reward):
        if reward > 0:
            return 0
        else:
            return 1

    def update_QTable(self, state, action_num, reward):
        # Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])
        self.qTable[state, action_num] = self.qTable[state, action_num] + self.learning_rate * (
                reward + self.discount_factor * max(self.qTable[state]) - self.qTable[state, action_num])

    def get_features(self, features):
        return features.std()

    def find_best_action(self, image, model):
        features = model.extract_features(image)
        print("features",features)
        m_before = self.get_features(features)
        print(m_before)
        for e in range(self.episodes):
            action_num = self.select_random_action()
            image_after = self.apply_action(action_num, image)
            features_after = model.extract_features(image_after.to("cuda"))
            m_after = self.get_features(features_after)
            reward = self.get_reward(m_before, m_after)
            state = self.get_state(reward)
            # print("------------------------------------")
            # print("Repeat: ", e)
            # print("State: ", state)
            # print("Action: ", action_num)
            # print("Reward: ", reward)
            # print("------------------------------------")
            self.update_QTable(state, action_num, reward)
        print("*************************************")
        print("Q-table")
        print(self.qTable)
        print("*************************************")

    def choose_final_action(self):
        return np.where(self.qTable == np.amax(self.qTable))[1][0]
