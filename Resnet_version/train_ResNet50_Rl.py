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

class_count = [1583, 4273]
w0 = (class_count[1]) / (class_count[0])
w1 = (class_count[1]) / (class_count[1])

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

            _, output_label_before = torch.max(metrics_before, dim=1)
            _, output_label_after = torch.max(metrics_after, dim=1)

            reward = rl_model.get_reward_according_to_label(m_before=metrics_before,
                                                            label_before=output_label_before,
                                                            m_after=metrics_after,
                                                            label_after=output_label_after,
                                                            label_target=label)

            current_q_value = q_values[0, action]

            # observed_q_value = reward + (gamma * next_q_value)
            observed_q_value = reward

            loss_value = (observed_q_value - current_q_value) ** 2

            # https://stackoverflow.com/questions/64856195/what-is-tape-based-autograd-in-pytorch
            loss_value.backward()
            rl_model.optimizer.step()
            rl_model.optimizer.zero_grad()

print("***************************************** Begin Validation *****************************************")

correct_rl = 0
total_rl = 0
more_confident = 0
correct_to_incorrect = 0

rl_model.model.eval()
for batch in dataloaders['val']:
    for image, label in zip(batch[0], batch[1]):
        image = image.to(device).to("cuda").unsqueeze(0)
        label = label.to(device).to("cuda")

        state = image

        outputs = rl_model.model(state)

        _, preds = torch.max(outputs, dim=1)

        action = preds.item()

        new_state = rl_model.apply_action(action, image).to("cuda")

        prediction_percent = class_model.extract_propabilities(new_state)
        _, prediction_after = torch.max(prediction_percent, dim=1)

        old_percent = class_model.extract_propabilities(image)

        _, output_label_before = torch.max(old_percent, dim=1)

        reward = rl_model.get_reward_according_to_label(m_before=old_percent,
                                                        label_before=output_label_before,
                                                        m_after=prediction_percent,
                                                        label_after=prediction_after,
                                                        label_target=label)

        if label > 0 and 0 < reward < 1:
            more_confident = more_confident + 1
        elif label < 0 and 0 < reward < 1:
            more_confident = more_confident + 1

        if prediction_after == label:
            correct_rl = correct_rl + 1

        if output_label_before == label and not(prediction_after == label):
            correct_to_incorrect = correct_to_incorrect + 1

        total_rl = total_rl + 1

print('\nMore confident: %2d%%  (%2d/%2d)' %
      (100. * more_confident / correct_rl, more_confident, correct_rl))

print(correct_to_incorrect)

print('\nVal Accuracy at rl only: %2d%% (%2d/%2d)' % (
    100. * correct_rl / total_rl, correct_rl, total_rl))
#
# weight_rl_dir = os.path.join('..', 'weights', 'train_rl_chestxray_dataset_extended.hdf5')
# torch.save(rl_model.model.state_dict(), weight_rl_dir)
# print("\nWeights saved successfully at: ", weight_rl_dir)
#
# dataloaders_dir = os.path.join('..', 'dataloaders', 'dataloaders_extended.pt')
# torch.save(dataloaders, dataloaders_dir)
# print("\nDataloaders saved successfully at: ", dataloaders_dir)

correct_rl = 0
total_rl = 0

print("**********************Everything Check*************************")
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

            prediction_after = class_model.test_image(new_state)

            if prediction_after == label:
                correct_rl = correct_rl + 1

            total_rl = total_rl + 1

print('\nTest Accuracy at rl only: %2d%% (%2d/%2d)' % (
    100. * correct_rl / total_rl, correct_rl, total_rl))

prediction = class_model.test_model(dataloaders['test'], device)

correct = 0
total = 0

print("**********************Only RL PREDICT*************************")
with torch.no_grad():
    rl_model.model.eval()
    for batch in dataloaders['test']:
        for image, label in zip(batch[0], batch[1]):
            image = image.to(device).to("cuda").unsqueeze(0)
            label = label.to(device).to("cuda")

            prediction_b4 = class_model.test_image(image)

            if not (prediction_b4 == label):
                state = image

                outputs = rl_model.model(state)

                action = preds.item()

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

print("\nOnly Classifier without RL prediction.")
prediction = class_model.test_model(dataloaders['test'], device)
