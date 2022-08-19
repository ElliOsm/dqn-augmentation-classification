from thesis.Resnet_version.resnet_dqn import resnetDqn
from thesis.data_prossesing.data_pytorch import data_reader
from thesis.data_prossesing.data_pytorch import get_default_device
from thesis.model.ResNet50_classifier import ResNet50

import os

# set device
device = get_default_device()
print("Device:", device)

rl_model = resnetDqn(batch_size=4)
rl_model.model.to(device)

feature_extractor = ResNet50()
feature_extractor.to(device)

# load dataset
# data_dir = '../data/train_i2a2_complete/data'
print("Dataset: CHESTXRAY")
data_dir = os.path.join('..', 'data', 'trainFolder', 'data')
dataloaders = data_reader(data_dir)
data = next(iter(dataloaders['test']))
image = data[0].to("cuda")
label = data[1].item()

correct = 0
total = 0

for e in range(rl_model.episodes):
    print('Epoch {}/{}'.format(e, rl_model.episodes))
    print('-' * 10)
    for item in dataloaders['test']:
        image = item[0].to("cuda")
        label = item[1].item()
        state = image
        action = rl_model.select_action(image)
        new_state = rl_model.apply_action(action, image).to("cuda")

        metrics_before = feature_extractor.extract_propabilities(image=state)
        metrics_after = feature_extractor.extract_propabilities(image=new_state)
        metrics_before_class0 = metrics_before[0][0].to("cpu").detach().numpy()
        metrics_before_class1 = metrics_before[0][1].to("cpu").detach().numpy()
        metrics_after_class0 = metrics_after[0][0].to("cpu").detach().numpy()
        metrics_after_class1 = metrics_after[0][1].to("cpu").detach().numpy()

        output_label_before = feature_extractor.get_classification_result(image=state).item()
        output_label_after = feature_extractor.get_classification_result(image=new_state).item()

        if output_label_before == 0:
            metrics_before_cc = metrics_before_class0
            metrics_after_cc = metrics_after_class0
        else:
            metrics_before_cc = metrics_before_class1
            metrics_after_cc = metrics_after_class1

        reward = rl_model.get_reward_according_to_label(m_before=metrics_before_cc,
                                                        label_before=output_label_before,
                                                        m_after=metrics_after_cc,
                                                        label_after=output_label_after,
                                                        label_target=label)

        q_values = rl_model.model(state)
        observed_q_value = reward

        loss_value = (observed_q_value - q_values)**2
        #https://stackoverflow.com/questions/64856195/what-is-tape-based-autograd-in-pytorch
        loss_value.sum().backward()
        rl_model.optimizer.step()
        rl_model.optimizer.zero_grad()

        if reward == 1:
            correct = correct +1
        total = total +1


print("outcome : " )
print(correct)
print(total)

# import torch
# weight_dir = os.path.join('..', 'weights', 'train_rl_minus_chestxray_dataset.hdf5')
# torch.save(rl_model.model.state_dict(), weight_dir)


