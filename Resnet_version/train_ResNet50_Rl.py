from thesis.Resnet_version.resnet_dqn import resnetDqn
from thesis.data_prossesing.data_pytorch import data_reader
from thesis.data_prossesing.data_pytorch import get_default_device
from thesis.model.ResNet50_classifier import ResNet50

import torch
import os

# set device
device = get_default_device()
print("Device:", device)

rl_model = resnetDqn(batch_size=4)
rl_model.model.to(device)

# load dataset
# data_dir = '../data/train_i2a2_complete/data'
print("Dataset: CHESTXRAY")
data_dir = os.path.join('..', 'data', 'trainFolder', 'data')
dataloaders = data_reader(data_dir)

class_model = ResNet50()
class_model.freeze()
class_model.to(device)

weight_dir = os.path.join('..', 'weights', 'train_chestxray_dataset.hdf5')
class_model.load_state_dict(torch.load(weight_dir))
print("Weights loaded successfully from: ", weight_dir)

torch.cuda.empty_cache()

print("\n***************************************** Begin Training *****************************************")
rl_model.model.train()
for batch in dataloaders['train']:
    for e in range(rl_model.episodes):
        for image, label in zip(batch[0], batch[1]):
            state = image.to(device).unsqueeze(0)
            label = label.to(device).item()

            q_values = rl_model.model(state)

            _, action = torch.max(q_values, dim=1)

            # Because it is returned in tensor and we need it as number
            action = action.item()

            new_state = rl_model.apply_action(action, image).to(device)

            metrics_before = class_model.extract_propabilities(image=state)
            metrics_after = class_model.extract_propabilities(image=new_state)

            _, prediction_before = torch.max(metrics_before, dim=1)
            _, prediction_after = torch.max(metrics_after, dim=1)

            reward = rl_model.get_reward(metrics_before, metrics_after, label)

            current_q_value = q_values[0, action]

            # observed_q_value = reward + (gamma * next_q_value)
            observed_q_value = reward

            loss_value = (observed_q_value - current_q_value) ** 2

            # https://stackoverflow.com/questions/64856195/what-is-tape-based-autograd-in-pytorch
            loss_value.backward()
            rl_model.optimizer.step()
            rl_model.optimizer.zero_grad()

print("\n***************************************** Begin Validation *****************************************")

correct_rl = 0
total_rl = 0
more_confident = 0
less_confident =0

rl_model.model.eval()
for batch in dataloaders['val']:
    for image, label in zip(batch[0], batch[1]):
        state = image.to(device).to("cuda").unsqueeze(0)
        label = label.to(device).to("cuda")

        outputs = rl_model.model(state)
        _, preds = torch.max(outputs, dim=1)

        action = preds.item()
        new_state = rl_model.apply_action(action, state).to("cuda")

        metrics_after = class_model.extract_propabilities(new_state)
        _, prediction_after = torch.max(metrics_after, dim=1)

        metrics_before = class_model.extract_propabilities(state)

        reward = rl_model.get_reward(metrics_before, metrics_after, label)

        if reward >= 0 :
            more_confident = more_confident + 1
        else:
            less_confident = less_confident +1

        if prediction_after == label:
            correct_rl = correct_rl + 1
        total_rl = total_rl + 1

print('More confident: %2d%%  (%2d/%2d)' %
      (100. * more_confident / correct_rl, more_confident, correct_rl))
print('Less confident: %2d%%  (%2d/%2d)' %
      (100. * less_confident / correct_rl, less_confident, correct_rl))

print('Val Accuracy at rl only: %2d%% (%2d/%2d)' % (
    100. * correct_rl / total_rl, correct_rl, total_rl))

weight_rl_dir = os.path.join('..', 'weights', 'train_rl_chestxray_dataset_deep.hdf5')
torch.save(rl_model.model.state_dict(), weight_rl_dir)
print("\nWeights saved successfully at: ", weight_rl_dir)

dataloaders_dir = os.path.join('..', 'dataloaders', 'dataloaders_deep.pt')
torch.save(dataloaders, dataloaders_dir)
print("\nDataloaders saved successfully at: ", dataloaders_dir)
