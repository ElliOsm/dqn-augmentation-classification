import pandas as pd
import numpy as np
import tensorflow.keras as k

from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.resnet50 import preprocess_input


def data_reader_augmentation(dir):
    image_directory = dir + '/images'
    data = pd.read_csv(dir + '/train.csv')

    # map = {
    #     0 : np.array([1,0]),
    #     1 : np.array([0,1])
    # }
    #
    # data["pneumonia"] = data["pneumonia"].apply(lambda x : map[x])
    #
    # print(data["pneumonia"].head)

    dataGenerator = ImageDataGenerator(rotation_range=40,
                                       width_shift_range=0.20,
                                       height_shift_range=0.20,
                                       zoom_range=0.20,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       validation_split=0.2,
                                       preprocessing_function=preprocess_input)

    trainGenerator = dataGenerator.flow_from_dataframe(data,
                                                       directory=image_directory,
                                                       x_col='fileName',
                                                       y_col='pneumonia',
                                                       target_size=(244, 244),
                                                       color_mode='rgb',
                                                       class_mode='raw',
                                                       shuffle=True,
                                                       batch_size=32,
                                                       subset='training')

    validationGenerator = dataGenerator.flow_from_dataframe(data,
                                                            directory=image_directory,
                                                            x_col='fileName',
                                                            y_col='pneumonia',
                                                            target_size=(244, 244),
                                                            color_mode='rgb',
                                                            class_mode='raw',
                                                            shuffle=True,
                                                            batch_size=32,
                                                            subset='validation')


    return trainGenerator,validationGenerator

def one_hot_encoding(x):
    return k.utils.to_categorical(x, 2)

def data_reader_evaluation(dir):
    image_directory = dir + '/images'
    data = pd.read_csv(dir + '/GTruth.csv')

    data['Id'] = data['Id'].apply(lambda x: str(x) + r".jpeg")
    data["Ground_Truth"] = data["Ground_Truth"].apply(lambda x: str(1 - x))

    dataGenerator = ImageDataGenerator( preprocessing_function=preprocess_input)

    evaluationGenerator = dataGenerator.flow_from_dataframe(data,
                                                            directory=image_directory,
                                                            x_col='Id',
                                                            y_col='Ground_Truth',
                                                            target_size=(244, 244),
                                                            class_mode='binary',
                                                            shuffle=True,
                                                            batch_size=32)

    return  evaluationGenerator