import os
import pandas as pd
import shutil



data_csv = pd.read_csv('../data/trainFolder/GTruth.csv')
image_dir = os.path.join('..','data','trainFolder', 'images')



data_csv['path'] = data_csv['Id'].map(lambda x: os.path.join(image_dir))
data_csv['path'] = data_csv['path'].astype(str).str.cat(data_csv[['Id']].astype(str), sep='/')


i = 0
for i in range(len(data_csv)):
    filenames = str(data_csv['Id'][i]) + '.jpeg'
    filelabel = data_csv['Ground_Truth'][i]
    filepath = data_csv['path'][i]+'.jpeg'

    if filelabel == 0:
        destination_healthy = os.path.join('..', 'data', 'trainFolder', 'data', '0', filenames)
        print(destination_healthy)
        print(filepath)
        shutil.copyfile(filepath, destination_healthy)
    elif filelabel == 1:
        destination_pneumonia = os.path.join('..', 'data', 'trainFolder', 'data', '1', filenames)
        print(destination_pneumonia)
        print(filepath)
        shutil.copyfile(filepath, destination_pneumonia)
    else:
        destination_test = os.path.join('..', 'data', 'train_i2a2_complete', 'data', 'test', filenames)
        shutil.copyfile(filepath, destination_test)

