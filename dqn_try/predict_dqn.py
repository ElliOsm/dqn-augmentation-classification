import os
from thesis.dqn_try.dqn import DQNAgent
from thesis.data_prossesing.data_pytorch import data_reader, get_default_device
from thesis.model.ResNet50_classifier import ResNet50

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

print("Label:")
print(label)


for e in range(agent.episodes):
    state = image.to("cuda")
    action = agent.select_action(image)
    new_state = agent.apply_action(action, image).to("cuda")
    metrics_before = classifier.extract_propabilities(image=state)
    metrics_after = classifier.extract_propabilities(image=new_state)
    metrics_before_numpy = metrics_before.to("cpu").detach().numpy()
    metrics_after_numpy = metrics_after.to("cpu").detach().numpy()
    output_label_before = classifier.get_classification_result(image=state)
    output_label_after = classifier.get_classification_result(image=new_state)
    reward = agent.get_reward_according_to_label(metrics_before_numpy, metrics_after_numpy,output_label_before,output_label_after, label)
    agent.train_model(image, action, reward)
    agent.target_net.load_state_dict(agent.policy_net.state_dict())
print('Complete')