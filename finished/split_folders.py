import os
import pandas as pd
import shutil


data_csv = pd.read_csv('../data/testFolder/train.csv')
image_dir = os.path.join('..','data','testFolder', 'images')

data_csv['path'] = data_csv['fileName'].map(lambda x: os.path.join(image_dir))
data_csv['path'] = data_csv['path'].str.cat(data_csv[['fileName']], sep='/')

i = 0
for i in range(len(data_csv)):
    filenames = data_csv['fileName'][i]
    filelabel = data_csv['pneumonia'][i]
    filepath = data_csv['path'][i]

    if filelabel == 0:
        destination_healthy = os.path.join('..', 'data', 'testFolder', 'data', '0', filenames)
        shutil.copyfile(filepath, destination_healthy)
    elif filelabel == 1:
        destination_pneumonia = os.path.join('..', 'data', 'testFolder', 'data', '1', filenames)
        shutil.copyfile(filepath, destination_pneumonia)
    else:
        destination_test = os.path.join('..', 'data', 'testFolder', 'data', 'test', filenames)
        shutil.copyfile(filepath, destination_test)

