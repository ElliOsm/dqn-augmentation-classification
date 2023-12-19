# Xray pneumonia classification using DQN and transfer learning 
## Origin
Idea for implementaion came from this paper: [Image Classification by Reinforcement Learning with Two-State Q-Learning](https://arxiv.org/abs/2007.01298)
## Description
Used a pretrained Resnet50 as a classifier

Used a DQN to find the appropriate transformation from a set of 3 actions:
adjust contrast,equalise and change sharpness,
in order to achieve better results at classifier

I also tried to add noise to the images to see if the results after the classifier would be better
## Dataset
My initial [dataset](https://www.kaggle.com/c/i2a2-brasil-pneumonia-classification/overview) consisted of 2 classes with xray images(pneumonia,healthy)

I also created a version with 7 classes with various sport images
