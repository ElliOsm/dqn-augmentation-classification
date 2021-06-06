import os
import logging
# Environment Variables

##Supresses tensorflow warnings and errors, change to 2 to show errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from thesis.old.ResNet50 import resnet50_compile
from thesis.old.data import data_reader_augmentation


train_datagen, val_datagen = data_reader_augmentation('../data/i2a2-brasil-pneumonia-classification')

model = resnet50_compile()

model.fit_generator(train_datagen,
                    epochs = 30,
                    #steps_per_epoch = 32,
                    validation_data=val_datagen,
                    #validation_steps= 32,
                    verbose=1)



model.save_weights("../weights/ResNet50_weights.hdf5")

