import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms



class DQNAgent:

    def __init__(self):
        self.actions = {
            '0': self.action1,
            '1': self.action2,
            '2': self.action3
        }
        self.qTable = np.zeros((2, 3)) #Q = np.zeros((state_size, action_size))
        self.state = [1, 2]
        self.learning_rate = 0.4
        self.discount_factor = 0.3

    # # rotate 90 left
    # def action1(self, image):
    #     unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #     unorm(image)
    #
    #     transform = transforms.Compose([
    #         transforms.ToPILImage()
    #     ])
    #     image = transform(image)
    #     image = transforms.functional.rotate(img=image, angle=90)
    #     transform = transforms.Compose([
    #         transforms.ToTensor()
    #     ])
    #
    #     return transform(image)
    #
    # # rotate 90 right
    # def action2(self, image):
    #     unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #     unorm(image)
    #
    #     transform = transforms.Compose([
    #         transforms.ToPILImage()
    #     ])
    #     image = transform(image)
    #     image = transforms.functional.rotate(img=image, angle=-90)
    #     transform = transforms.Compose([
    #         transforms.ToTensor()
    #     ])
    #
    #     return transform(image)
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
        transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        image = transform(image)
        image = transforms.functional.adjust_contrast(image, contrast_factor=3)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        return transform(image)

    #equalize image
    def action2(self, image):
        transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        image = transform(image)
        image = transforms.functional.equalize(img=image)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        return transform(image)

    # sharpness
    def action3(self,image):

        transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        image = transform(image)
        image = transforms.functional.adjust_sharpness(img=image, sharpness_factor=10)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        return transform(image)


    def select_random_action(self):
        return random.randint(0, 2)

    def apply_action(self, action_num, image):
        action = self.actions[action_num]
        image = action(image)
        return  image

    def get_reward(self,m_before, m_after):
        difference = m_after - m_before
        if difference > 0:
            return 1
        elif difference == 0:
            return 0
        else:
            return -1

    def get_state(self, reward):
        return 0 if reward > 0 else 1

    def update_QTable(self, state, action_num, reward):
        #Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])
        self.qTable[state][action_num] = self.qTable[state][action_num] + self.learning_rate * (
                reward + self.discount_factor * '''self.qTable[state + 1]''' - self.qTable[state][action_num])

    #TODO:Find Best Action
    def find_best_action(self,image):
        # TODO: m_before
        num_episodes = 3*10
        for e in range(num_episodes):
            action_num = self.select_random_action()
            image_after = self.apply_action(action_num,image)
            #TODO:find m_after
            reward = self.get_reward(m_before=,
                                     m_after=)
            state = self.get_state(reward)
            self.update_QTable(state,action_num,reward)


    #TODO:predict