from thesis.Resnet_version.resnet_dqn import resnetDqn
from thesis.data_prossesing.data_pytorch import data_reader
from thesis.data_prossesing.data_pytorch import get_default_device
from thesis.model.ResNet50_classifier import ResNet50
import torch
import torch.nn as nn

import os

# set device
device = get_default_device()
print("Device:", device)

rl_model = resnetDqn(batch_size=4)
rl_model.model.to(device)

class_model = ResNet50()
class_model.freeze()
class_model.to(device)

weight_dir = os.path.join('..', 'weights', 'train_chestxray_dataset.hdf5')
class_model.load_state_dict(torch.load(weight_dir))
print("Weights for classification model loaded successfully from: ", weight_dir)

weight_rl_dir = os.path.join('..', 'weights', 'train_rl_chestxray_dataset_deep.hdf5')
rl_model.model.load_state_dict(torch.load(weight_rl_dir))
print("weights for reinforcement model loaded successfully from: ", weight_dir)

print("Dataset: CHESTXRAY")
dataloaders_dir = os.path.join('..', 'dataloaders', 'dataloaders_deep.pt')
dataloaders = torch.load(dataloaders_dir)

print("\n***************************************** Predict Scenarios *****************************************")
print("\n**Everything goes through RL and then classified.")

correct_rl = 0
total_rl = 0
more_confident = 0
less_confident = 0

with torch.no_grad():
    rl_model.model.eval()
    for batch in dataloaders['test']:
        for image, label in zip(batch[0], batch[1]):
            image = image.to(device).to("cuda").unsqueeze(0)
            label = label.to(device).to("cuda")
            state = image

            outputs = rl_model.model(state)
            _, preds = torch.max(outputs, dim=1)

            action = preds.item()

            new_state = rl_model.apply_action(action, image).to("cuda")

            propabilities_after = class_model.extract_propabilities(new_state)
            _, prediction_after = torch.max(propabilities_after, dim=1)

            propabilities_before = class_model.extract_propabilities(image)
            _, prediction_before = torch.max(propabilities_before, dim=1)

            reward = rl_model.get_reward(propabilities_before, propabilities_after, label)

            more_confident, less_confident = rl_model.getConfidence(label,
                                                                    propabilities_after,
                                                                    propabilities_before,
                                                                    more_confident,
                                                                    less_confident)

            prediction_after = prediction_after.item()
            prediction_before = prediction_before.item()
            label = label.item()

            if prediction_after == label:
                correct_rl = correct_rl + 1

            total_rl = total_rl + 1

print('Test Accuracy: %2d%% (%2d/%2d)' % (
    100. * correct_rl / total_rl, correct_rl, total_rl))

print('More confident: %2d%%  (%2d/%2d)' %
      (100. * more_confident / correct_rl, more_confident, correct_rl))
print('Less confident: %2d%%  (%2d/%2d)' %
      (100. * less_confident / correct_rl, less_confident, correct_rl))

print("\n**Only incorrect predictions are fed into RL and then Re-classified.")

correct = 0
total = 0
correct_rl = 0
total_rl = 0
correct_to_incorrect = 0

with torch.no_grad():
    rl_model.model.eval()
    for batch in dataloaders['test']:
        for image, label in zip(batch[0], batch[1]):
            image = image.to(device).to("cuda").unsqueeze(0)
            label = label.to(device).to("cuda")

            prediction_b4 = class_model.test_image(image)

            if prediction_b4 != label:
                state = image

                outputs = rl_model.model(state)
                _, preds = torch.max(outputs, dim=1)
                action = preds.item()

                new_state = rl_model.apply_action(action, image).to("cuda")

                prediction_after = class_model.test_image(new_state)

                if prediction_b4 == label:
                    if prediction_after != label:
                        correct_to_incorrect = correct_to_incorrect + 1

                if prediction_after == label:
                    correct_rl = correct_rl + 1
                    correct = correct + 1

                total_rl = total_rl + 1
            else:
                correct = correct + 1

            total = total + 1

print('Test Accuracy: %2d%% (%2d/%2d)' % (
    100. * correct / total, correct, total))

print('Test Accuracy at rl only: %2d%% (%2d/%2d)' % (
    100. * correct_rl / total_rl, correct_rl, total_rl))

print('Corrects that were changed to incorrect:', correct_to_incorrect)

print("\n**Only Classifier without RL prediction.")
prediction = class_model.test_model(dataloaders['test'], device)

print("\n**Predictions with noise")

correct_rl_with_noise = 0
total_rl_with_noise = 0
more_confident = 0
less_confident = 0

with torch.no_grad():
    rl_model.model.eval()
    for batch in dataloaders['test']:
        for image, label in zip(batch[0], batch[1]):
            image = image.to(device).to("cuda").unsqueeze(0)
            label = label.to(device).to("cuda")
            state = image

            outputs = rl_model.model(state)
            _, preds = torch.max(outputs, dim=1)

            action = preds.item()

            new_state = rl_model.apply_action(action, image).to("cuda")

            new_state = rl_model.add_noise(new_state).to("cuda")

            new_state = new_state.float()

            propabilities_after = class_model.extract_propabilities(new_state)
            _, prediction_after = torch.max(propabilities_after, dim=1)

            propabilities_before = class_model.extract_propabilities(image)
            _, prediction_before = torch.max(propabilities_before, dim=1)

            reward = rl_model.get_reward(propabilities_before, propabilities_after, label)


            prediction_after = prediction_after.item()
            label = label.item()

            if prediction_after == label:
                correct_rl_with_noise = correct_rl_with_noise + 1

            total_rl_with_noise = total_rl_with_noise + 1

print('Test Accuracy: %2d%% (%2d/%2d)' % (
    100. * correct_rl_with_noise / total_rl_with_noise, correct_rl_with_noise, total_rl_with_noise))
