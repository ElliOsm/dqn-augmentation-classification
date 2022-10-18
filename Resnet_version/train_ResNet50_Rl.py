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

weight_rl_dir = os.path.join('..', 'weights', 'train_rl_chestxray_dataset.hdf5')
rl_model.model.load_state_dict(torch.load(weight_rl_dir))
print("Weights loaded successfully from: ", weight_rl_dir)

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

            #Because it is returned in tensor and we need it as number
            action = action.item()

            new_state = rl_model.apply_action(action, image).to(device)

            metrics_before = class_model.extract_propabilities(image=state)
            metrics_after = class_model.extract_propabilities(image=new_state)

            _, output_label_before = torch.max(metrics_before, dim=1)
            _, output_label_after = torch.max(metrics_after, dim=1)

            # metrics_before_class0 = metrics_before[0][0].to("cpu").detach().numpy()
            # metrics_before_class1 = metrics_before[0][1].to("cpu").detach().numpy()
            # metrics_after_class0 = metrics_after[0][0].to("cpu").detach().numpy()
            # metrics_after_class1 = metrics_after[0][1].to("cpu").detach().numpy()

            # if output_label_before == 0:
            #     metrics_before_cc = metrics_before_class0
            #     metrics_after_cc = metrics_after_class0
            # else:
            #     metrics_before_cc = metrics_before_class1
            #     metrics_after_cc = metrics_after_class1

            reward = rl_model.get_reward_according_to_label(m_before=metrics_before,
                                                            label_before=output_label_before,
                                                            m_after=metrics_after,
                                                            label_after=output_label_after,
                                                            label_target=label)

            q_value = q_values[0, action]
            observed_q_value = reward

            print("Q-value: ", q_value)
            print("Observed Q-value: ", observed_q_value)

            loss_value = (class_weights * (observed_q_value - q_value) ** 2).mean()
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

# weight_rl_dir = os.path.join('..', 'weights', 'train_rl_chestxray_dataset.hdf5')
# torch.save(rl_model.model.state_dict(), weight_rl_dir)
# print("Weights saved successfully at: ", weight_rl_dir)
