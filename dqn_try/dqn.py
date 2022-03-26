import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from thesis.dqn_try.buffer import ReplayMemory
from thesis.dqn_try.dqn_architectuer import DQN
from thesis.model.ResNet50_classifier import ResNet50


class DQNAgent:

    def __init__(self, batch_size, dir=None):
        self.actions = {
            0: self.action1,
            1: self.action2,
            2: self.action3
        }
        self.action_space = len(self.actions)

        self.learning_rate = 0.4
        self.discount = 0.3
        self.episodes = 30
        self.exploration = 1
        self.exploration_min = 0.1

        self.policy_net = DQN().to("cuda")
        self.target_net = DQN().to("cuda")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.buffer = ReplayMemory(10000)

        self.batch_size = batch_size
        self.steps_done = 0

        self.threshold = 0.80

        self.classifier = ResNet50(pretrained=True)
        if dir is not None:
            self.classifier.load_state_dict(torch.load(dir))
            print("weights loaded successfully from: ", dir)

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

    # adjust_brightness
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
        # define epsilon
        if self.steps_done < 10:
            epsilon = self.exploration
        else:
            epsilon = self.exploration_min
        self.steps_done += 1

        # https://stackoverflow.com/questions/33359740/random-number-between-0-and-1-in-python
        if np.random.uniform(0, 1) < epsilon:
            print("random")
            action_num = random.randint(0, 2)
        else:
            print("policy net")
            action_num = self.policy_net(image)
            _, action_num = torch.max(action_num, dim=1)
            action_num = action_num.item()

        return action_num

    def apply_action(self, action_num, image):
        action = self.actions[action_num]
        image = action(image)
        return image

    def get_reward(self, m_after, target_label):
        difference = (target_label - m_after).float()
        if difference < self.threshold:
            return 1
        elif difference == 0:
            return 0
        else:
            return -1

    def get_reward_according_to_label(self,m_before, m_after, target):
        difference_before = (m_before - target).float()
        difference_after = (m_after - target).float()
        if difference_after > difference_before:
            return 1
        elif difference_after == difference_before:
            return 0
        else:
            return -1

    def get_Q_value(self, state):
        q_values = self.policy_net(state)
        return q_values

    # def training(self):
    #     if len(self.buffer) < self.batch_size:
    #         return
    #
    #     state, action_num, reward, next_state, done = self.buffer.sample(self.batch_size)
    #
    #     state = state.pop(0).to(self.device)
    #     action_num = action_num.pop(0)
    #     reward = reward.pop(0)
    #     next_state = next_state.pop(0)
    #     # done = done.pop(0)
    #
    #     q_eval = self.policy_net.test(state)
    #     q_next = self.target_net(next_state).detach()
    #     q_target = reward + self.discount * q_next.max(1)[0].view(self.batch_size, 1)
    #     loss = self.loss_func(q_eval, q_target)
    #
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    # def train(self, image, reward):
    #     q_value = self.policy_net(image)
    #     image_next = image
    #     action_num = self.select_action(image_next)
    #     image_next = self.apply_action(action_num, image_next)
    #     next_q_value = self.policy_net(image_next)
    #
    #     q_value[action_num] = reward + self.discount * np.amax(next_q_value)

    # def train_model(self, image, action, reward, new_image):
    #     old_q_values = self.get_Q_value(image)
    #     new_q_values = self.get_Q_value(new_image)
    #
    #     old_q_values[action] = reward + self.discount * np.amax(new_q_values)
    #
    #     training_data = [[image]]
    #     target_output = [old_q_values]
    #
    #     training_data = {self.model_input: training_data, self.target_output: target_output}
    #
    #     self.policy_net(self.optimizer)


    def train_model(self,image, action, reward):
        old_q_values = self.get_Q_value(image)
        new_image = self.apply_action(action,image)
        new_q_values = self.get_Q_value(new_image)
        expected_values = reward + self.discount * np.amax(new_q_values)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(old_q_values, expected_values)

        # Optimize the model
        self.policy_net.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_net.optimizer.step()

    def train_loop(self,image,label):
        for e in range(self.episodes):
            state = image
            action = self.select_action(image)
            new_state = self.apply_action(action,image)
            metrics_before = self.target_net(state)
            metrics_after = self.target_net(new_state)
            reward = self.get_reward_according_to_label(metrics_before,metrics_after,label)
            self.train_model(image,action,reward)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        print('Complete')

    # def get_features(self, features):
    #     return features.std()

    # https://stackoverflow.com/questions/57727372/how-do-i-get-the-value-of-a-tensor-in-pytorch
    def get_difference(self, prob):
        prob_array = prob.cpu().detach().numpy()
        prob_array_copy = prob_array.copy()
        prob0 = prob_array_copy[0][0]
        prob1 = prob_array_copy[0][1]
        return abs((prob0) - (prob1))


    def predict_difference(self, image, model, target_label):
        probs = model.extract_propabilities(image).to("cuda")
        m_before = self.get_difference(probs)
        for e in range(self.episodes):
            action_num = self.select_random_action()
            image_after = self.apply_action(action_num, image)
            probs_after = model.extract_features(image_after.to("cuda"))
            m_after = self.get_features(probs_after)
            print(m_after)
            reward = self.get_reward(m_after, target_label)
            # state_num = self.get_state(reward)
            # if state_num == 0:
            state = image
            next_state = image_after
            # else:
            #     state = image_after
            #     next_state = image
            # done, next_state = self.is_done(next_state)
            self.training()
