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

weight_dir = os.path.join('..', 'weights', 'train_rl_chestxray_dataset.hdf5')
rl_model.model.load_state_dict(torch.load(weight_dir))
print("weights for reinforcement model loaded successfully from: ", weight_dir)

print("Dataset: CHESTXRAY")
data_dir = os.path.join('..', 'data', 'trainFolder', 'data')
dataloaders = data_reader(data_dir)

correct = 0
total = 0

correct_rl = 0
total_rl = 0

with torch.no_grad():
    rl_model.model.eval()
    for batch in dataloaders['train']:
        for image, label in zip(batch[0], batch[1]):
            image = image.to(device).to("cuda").unsqueeze(0)
            label = label.to(device).to("cuda")

            prediction_b4 = class_model.test_image(image)

            if not (prediction_b4 == label):
                state = image

                outputs = rl_model.model(state)
                probs = nn.Softmax(dim=1)
                outputs = probs(outputs)
                _, preds = torch.max(outputs, dim=1)

                action = preds.item()

                print(action)

                new_state = rl_model.apply_action(action, image).to("cuda")

                prediction_after = class_model.test_image(new_state)

                if prediction_after == label:
                    correct_rl = correct_rl + 1
                    correct = correct + 1

                total_rl = total_rl + 1
            else:
                correct = correct + 1

            total = total + 1

print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
    100. * correct / total, correct, total))

print('\nTest Accuracy at rl only: %2d%% (%2d/%2d)' % (
    100. * correct_rl / total_rl, correct_rl, total_rl))
