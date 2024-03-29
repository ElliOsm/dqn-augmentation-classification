from thesis.Resnet_version.ResNet50_Rl import ResNet50Rl
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

        self.learning_rate = 0.01
        self.discount = 0.3
        self.episodes = 10
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
        image = transforms.functional.adjust_contrast(image, contrast_factor=1.5)
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
        image = transforms.functional.adjust_brightness(img=image, brightness_factor=1.2)
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
        image = transforms.functional.adjust_sharpness(img=image, sharpness_factor=15)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        return image

    def apply_action(self, action_num, image):
        action = self.actions[action_num]
        image = action(image)
        return image

    def add_noise(self, image):
        mean = 0
        var = 10
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, (224, 224, 3))

        image = np.squeeze(image)
        transform = transforms.Compose([
            transforms.ToPILImage()
        ])
        image = transform(image)

        noisy_image = image + gaussian
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image = transform(noisy_image)
        image = torch.unsqueeze(image, 0)

        return image

    # def get_reward_according_to_label(self,
    #                                   m_before, label_before,
    #                                   m_after, label_after,
    #                                   label_target):
    #     if label_target == 0:
    #         label_target_diff = -1
    #     else:
    #         label_target_diff = 1
    #
    #     label_after = label_after.item()
    #     label_before = label_before.item()
    #
    #     if label_after == label_target:
    #         if label_before == label_after:
    #             m_before_cc = m_before[0][label_target].to("cpu").detach().numpy()
    #             m_after_cc = m_after[0][label_target].to("cpu").detach().numpy()
    #             difference = abs(m_after_cc - m_before_cc)
    #             if abs(label_target_diff - m_after_cc) < abs(label_target_diff - m_before_cc):
    #                 return difference
    #             else:
    #                 difference = -difference
    #                 return difference
    #         else:
    #             return 1
    #     else:
    #         if label_before == label_after:
    #             m_before_cc = m_before[0][label_after].to("cpu").detach().numpy()
    #             m_after_cc = m_after[0][label_after].to("cpu").detach().numpy()
    #             difference = abs(m_after_cc - m_before_cc)
    #             if abs(label_target_diff - m_after_cc) > abs(label_target_diff - m_before_cc):
    #                 return difference
    #             else:
    #                 difference = -difference
    #                 return difference
    #         else:
    #             return -1

    def get_reward(self, m_before, m_after, label_target):
        m_before_cc = m_before[0][label_target].to("cpu").detach().numpy()
        m_after_cc = m_after[0][label_target].to("cpu").detach().numpy()
        difference = m_after_cc - m_before_cc
        return difference

    def getConfidence(self, label, m_after, m_before, more_confident, less_confident):
        if label == 0:
            label_target_diff = -1
        else:
            label_target_diff = 1

        temp_before = label_target_diff - m_before[0][label].to("cpu").detach().numpy()
        temp_after = label_target_diff - m_after[0][label].to("cpu").detach().numpy()

        if abs(temp_after - temp_before) > 0:
             more_confident += 1
        else:
            less_confident += 1

        return more_confident,less_confident
