import os
from thesis.dqn_try.dqn import DQNAgent
from thesis.data_prossesing.data_pytorch import data_reader, get_default_device
from thesis.model.ResNet50_classifier import ResNet50
import numpy as np

# set device
device = get_default_device()
print("Device:", device)

# load dataset
# data_dir = '../data/train_i2a2_complete/data'
data_dir = os.path.join('..', 'data', 'trainFolder', 'data')
dataloaders = data_reader(data_dir)
data = next(iter(dataloaders['test']))
image = data[0].to("cuda")
label = data[1].item()
agent = DQNAgent(batch_size=4)

classifier = ResNet50()
classifier.freeze()
classifier.to("cuda")

for e in range(agent.episodes):
    state = image.to("cuda")
    action = agent.select_action(image)
    new_state = agent.apply_action(action, image).to("cuda")

    metrics_before = classifier.extract_propabilities(image=state)
    metrics_before = classifier.extract_propabilities(image=state)
    metrics_after = classifier.extract_propabilities(image=new_state)
    metrics_before_class0 = metrics_before[0][0].to("cpu").detach().numpy()
    metrics_before_class1 = metrics_before[0][1].to("cpu").detach().numpy()
    metrics_after_class0 = metrics_after[0][0].to("cpu").detach().numpy()
    metrics_after_class1 = metrics_after[0][1].to("cpu").detach().numpy()
    output_label_before = classifier.get_classification_result(image=state).item()
    output_label_after = classifier.get_classification_result(image=new_state).item()
    if output_label_before == 0:
        metrics_before_cc = metrics_before_class0
        metrics_after_cc = metrics_after_class0
    else:
        metrics_before_cc = metrics_before_class1
        metrics_after_cc = metrics_after_class1
    print(metrics_before_cc, metrics_after_cc, output_label_before,
          output_label_after, label)
    reward = agent.get_reward_according_to_label(m_before=metrics_before_cc,
                                                 label_before=output_label_before,
                                                 m_after=metrics_after_cc,
                                                 label_after=output_label_after,
                                                 label_target=label)
    print(metrics_before_cc, metrics_after_cc)
    print(reward)
    print("IMAGE BEFORE:", state.size())
    print("IMAGE AFTER:", new_state.size())
    agent.train_model(image, action, reward)
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
print('Complete')