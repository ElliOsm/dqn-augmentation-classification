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

correct = 0
total = 0

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

class_weights = torch.FloatTensor([w0, w1]).to(device)
print("class_weights: ", class_weights)

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

            q_value = q_values[0, action]
            observed_q_value = reward

            print("Q-value: ", q_value)
            print("Observed Q-value: ", observed_q_value)

            loss_value = (observed_q_value - q_value) ** 2
            print(loss_value)

            # https://stackoverflow.com/questions/64856195/what-is-tape-based-autograd-in-pytorch
            loss_value.backward()
            rl_model.optimizer.step()
            rl_model.optimizer.zero_grad()

            print("Action_choosen:", action)
            print("Reward:", reward)

    print("Classification After.")
    print("----------------------------------------------------")
    result = class_model.test_image(state)
    print("Predicted Class: ", result)
    print("True Class: ", label)
    print("----------------------------------------------------")

    if result == label:
        correct = correct + 1
    total = total + 1

print('\nTrain Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))

correct_rl = 0
total_rl = 0
for batch in dataloaders['val']:
    for image, label in zip(batch[0], batch[1]):
        image = image.to(device).to("cuda").unsqueeze(0)
        label = label.to(device).to("cuda")

        prediction_b4 = class_model.test_image(image)

        if not (prediction_b4 == label):
            state = image

            outputs = rl_model.model(state)
            _, preds = torch.max(outputs, dim=1)

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

print('\nVal Accuracy: %2d%% (%2d/%2d)' % (
    100. * correct / total, correct, total))

print('\nVal Accuracy at rl only: %2d%% (%2d/%2d)' % (
    100. * correct_rl / total_rl, correct_rl, total_rl))

weight_rl_dir = os.path.join('..', 'weights', 'train_rl_chestxray_dataset.hdf5')
torch.save(rl_model.model.state_dict(), weight_rl_dir)
print("Weights saved successfully at: ", weight_rl_dir)
