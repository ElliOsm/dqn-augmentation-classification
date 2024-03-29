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
        self.episodes = 5
        self.exploration = 1
        self.exploration_min = 0.1
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

    def select_random_action(self):
        return random.randint(0, 2)

    def select_action(self):
        # # define epsilon
        # if self.steps_done < 10:
        #     epsilon = self.exploration
        # else:
        #     epsilon = self.exploration_min
        # self.steps_done += 1
        #
        # # https://stackoverflow.com/questions/33359740/random-number-between-0-and-1-in-python
        # if np.random.uniform(0, 1) < epsilon:
        #     print("random")
        #     action_num = random.randint(0, 2)
        # else:
        #     print("qtable")
        #     max_value = np.amax(self.qTable.all(axis=1))
        #     for i in self.qTable[0,]:
        #         if self.qTable[0,i] == max_value:
        #             action_num = i
        #
        # return action_num

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
        # Q[state, action] = Q[state, action] + lr * (reward + discount * np.max(Q[new_state, :]) — Q[state, action])
        self.qTable[state][action_num] = self.qTable[state][action_num] + self.learning_rate * (
                reward + self.discount_factor * max(self.qTable[state]) - self.qTable[state][action_num])

    def get_features(self, features):
        return features.std()

    # https://stackoverflow.com/questions/57727372/how-do-i-get-the-value-of-a-tensor-in-pytorch
    def get_difference(self, prob):
        prob_array = prob.cpu().detach().numpy()
        prob_array_copy = prob_array.copy()
        prob0 = prob_array_copy[0][0]
        prob1 = prob_array_copy[0][1]
        return abs((prob0) - (prob1))

    def find_best_action_prob(self, image, model):
        probs = model.extract_propabilities(image).to("cuda")
        print("probs", probs)
        m_before = self.get_difference(probs)
        print(m_before)
        for e in range(self.episodes):
            action_num = self.select_random_action()
            image_after = self.apply_action(action_num, image)
            probs_after = model.extract_features(image_after.to("cuda"))
            m_after = self.get_features(probs_after)
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
        print("*************************************")        # print("*************************************")
        print("Q-table")
        print(self.qTable)
        print("*************************************")

    # def find_best_action(self, image, model):
    #     features = model.extract_features(image)
    #     print("features", features)
    #     m_before = self.get_features(features)
    #     print(m_before)
    #     for e in range(self.episodes):
    #         action_num = self.select_random_action()
    #         image_after = self.apply_action(action_num, image)
    #         features_after = model.extract_features(image_after.to("cuda"))
    #         m_after = self.get_features(features_after)
    #         reward = self.get_reward(m_before, m_after)
    #         state = self.get_state(reward)
    #         # print("------------------------------------")
    #         # print("Repeat: ", e)
    #         # print("State: ", state)
    #         # print("Action: ", action_num)
    #         # print("Reward: ", reward)
    #         # print("------------------------------------")
    #         self.update_QTable(state, action_num, reward)
    #     print("*************************************")
    #     print("Q-table")
    #     print(self.qTable)
    #     print("*************************************")

    def choose_final_action(self):
        return np.where(self.qTable == np.amax(self.qTable))[1][0]
