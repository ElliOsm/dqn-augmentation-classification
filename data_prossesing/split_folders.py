import os
import pandas as pd
import shutil

#comment in for i2a3-brasil-pneumonia-classification dataset
#
# data_csv = pd.read_csv('../data/train_i2a2_complete/train.csv')
# image_dir = os.path.join('..','data','train_i2a2_complete', 'images')
#
# data_csv['path'] = data_csv['fileName'].map(lambda x: os.path.join(image_dir))
# data_csv['path'] = data_csv['path'].str.cat(data_csv[['fileName']], sep='/')
#
# i = 0
# for i in range(len(data_csv)):
#     filenames = data_csv['fileName'][i]
#     filelabel = data_csv['pneumonia'][i]
#     filepath = data_csv['path'][i]
#
#     if filelabel == 0:
#         destination_healthy = os.path.join('..', 'data', 'train_i2a2_complete', 'data', '0', filenames)
#         shutil.copyfile(filepath, destination_healthy)
#     elif filelabel == 1:
#         destination_pneumonia = os.path.join('..', 'data', 'train_i2a2_complete', 'data', '1', filenames)
#         shutil.copyfile(filepath, destination_pneumonia)
#     else:
#         destination_test = os.path.join('..', 'data', 'train_i2a2_complete', 'data', 'test', filenames)
#         shutil.copyfile(filepath, destination_test)
#

#comment in for Pneumonia-Chest-X-ray dataset
# data_dir = os.path.join('..', 'data', 'trainFolder', 'GTruth.csv')
# data_csv = pd.read_csv(data_dir)
# image_dir = os.path.join('..', 'data', 'trainFolder', 'images')
#
# data_csv['path'] = data_csv['Id'].map(lambda x: os.path.join(image_dir))
# data_csv['path'] = data_csv['path'].astype(str).str.cat(data_csv[['Id']].astype(str), sep='/')
#
# i = 0
# for i in range(len(data_csv)):
#     filenames = str(data_csv['Id'][i]) + '.jpeg'
#     filelabel = data_csv['Ground_Truth'][i]
#     filepath = data_csv['path'][i] + '.jpeg'
#
#     if filelabel == 0:
#         destination_healthy = os.path.join('..', 'data', 'trainFolder', 'data', '0', filenames)
#         print(destination_healthy)
#         print(filepath)
#         shutil.copyfile(filepath, destination_healthy)
#     elif filelabel == 1:
#         destination_pneumonia = os.path.join('..', 'data', 'trainFolder', 'data', '1', filenames)
#         print(destination_pneumonia)
#         print(filepath)
#         shutil.copyfile(filepath, destination_pneumonia)
#     else:
#         destination_test = os.path.join('..', 'data', 'train_i2a2_complete', 'data', 'test', filenames)
#         shutil.copyfile(filepath, destination_test)


#comment in for sports dataset
# data_dir = os.path.join('..', 'data', 'sports-dataset', 'train.csv')
# data_csv = pd.read_csv(data_dir)
# image_dir = os.path.join('..','data','sports-dataset', 'train')
#
# data_csv['path'] = data_csv['image_ID'].map(lambda x: os.path.join(image_dir))
# data_csv['path'] = data_csv['path'].str.cat(data_csv[['image_ID']], sep='/')
#
# i = 0
# for i in range(len(data_csv)):
#     filenames = data_csv['image_ID'][i]
#     filelabel = data_csv['label'][i]
#     filepath = data_csv['path'][i]
#
#     if filelabel == 'Cricket':
#         destination_healthy = os.path.join('..', 'data', 'sports_complete', 'data', '0', filenames)
#         shutil.copyfile(filepath, destination_healthy)
#     elif filelabel == 'Wrestling':
#         destination_pneumonia = os.path.join('..', 'data', 'sports_complete', 'data', '1', filenames)
#         shutil.copyfile(filepath, destination_pneumonia)
#     elif filelabel == 'Karate':
#         destination_pneumonia = os.path.join('..', 'data', 'sports_complete', 'data', '2', filenames)
#         shutil.copyfile(filepath, destination_pneumonia)
#     elif filelabel == 'Tennis':
#         destination_pneumonia = os.path.join('..', 'data', 'sports_complete', 'data', '3', filenames)
#         shutil.copyfile(filepath, destination_pneumonia)
#     elif filelabel == 'Badminton':
#         destination_pneumonia = os.path.join('..', 'data', 'sports_complete', 'data', '4', filenames)
#         shutil.copyfile(filepath, destination_pneumonia)
#     elif filelabel == 'Soccer':
#         destination_pneumonia = os.path.join('..', 'data', 'sports_complete', 'data', '5', filenames)
#         shutil.copyfile(filepath, destination_pneumonia)
#     elif filelabel == 'Swimming':
#         destination_pneumonia = os.path.join('..', 'data', 'sports_complete', 'data', '6', filenames)
#         shutil.copyfile(filepath, destination_pneumonia)
#     else:
#         destination_test = os.path.join('..', 'data', 'sports_complete', 'data', 'unidentified', filenames)
#         shutil.copyfile(filepath, destination_test)
