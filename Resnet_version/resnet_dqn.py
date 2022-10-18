from ResNet50_Rl import ResNet50Rl
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


class resnetDqn:

    def __init__(self, batch_size):
        self.actions = {
            0: self.action1,
            1: self.action2,
            2: self.action3
        }
        self.action_space = len(self.actions)

        self.learning_rate = 0.001
        self.discount = 0.3
        self.episodes = 30
        self.exploration = 1
        self.exploration_min = 0.1

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = ResNet50Rl()
        self.model.to(self.device)

        self.batch_size = batch_size
        self.steps_done = 0

        self.threshold = 0.02

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

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

    # def select_action(self, image):
    #     # define epsilon
    #     if self.steps_done < 10:
    #         epsilon = self.exploration
    #     else:
    #         epsilon = self.exploration_min
    #     self.steps_done += 1
    #
    #     # https://stackoverflow.com/questions/33359740/random-number-between-0-and-1-in-python
    #     if np.random.uniform(0, 1) < epsilon:
    #         print("Choose an action at random.(exploration)")
    #         action_num = random.randint(0, 2)
    #     else:
    #         print("Choose an action according to policy net.(exploitation)")
    #         action_num = self.model(image)
    #         _, action_num = torch.max(action_num, dim=1)
    #         action_num = action_num.item()
    #
    #     return action_num

    def apply_action(self, action_num, image):
        action = self.actions[action_num]
        image = action(image)
        return image

    def get_reward_according_to_label(self,
                                      m_before, label_before,
                                      m_after, label_after,
                                      label_target):
        if label_after == label_target:
            if label_before == label_after:
                m_before_cc = m_before[0][label_target].to("cpu").detach().numpy()
                m_after_cc = m_after[0][label_target].to("cpu").detach().numpy()
                # difference = m_after_cc - m_before_cc
                if abs(m_after_cc)>abs(m_before_cc):
                    return 1
                else:
                    return -1
            else:
                return 1
        else:
            if label_before == label_after:
                m_before_cc = m_before[0][label_after].to("cpu").detach().numpy()
                m_after_cc = m_after[0][label_after].to("cpu").detach().numpy()
                if abs(m_after_cc) < abs(m_before_cc):
                    return 1
                else:
                    return -1
            else:
                return -1
