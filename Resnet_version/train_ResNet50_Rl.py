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


for e in range(rl_model.episodes):
    state = image.to("cuda")
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
    print(metrics_before_cc, metrics_after_cc, output_label_before,
          output_label_after, label)
    reward = rl_model.get_reward_according_to_label(m_before=metrics_before_cc,
                                                 label_before=output_label_before,
                                                 m_after=metrics_after_cc,
                                                 label_after=output_label_after,
                                                 label_target=label)
    print(metrics_before_cc, metrics_after_cc)
    print(reward)
    print("IMAGE BEFORE:", state.size())
    print("IMAGE AFTER:", new_state.size())
    # model.train_model(image, action, reward)
    # model.target_net.load_state_dict(model.policy_net.state_dict())
print('Complete')

