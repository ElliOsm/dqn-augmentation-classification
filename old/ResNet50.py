from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,GlobalMaxPooling2D
import tensorflow as tf



def resnet50_compile():


    pretrained_model = ResNet50(include_top=False,
                                classes=2,
                                weights='imagenet',
                                input_shape=(224,224,3))

    model = Sequential()
    model.add(pretrained_model)
    model.add(GlobalMaxPooling2D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.00001)

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.summary()

    return model


