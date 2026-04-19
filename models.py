import os.path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.src.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomTranslation
from keras.src.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Dense, Input, BatchNormalization, Dropout, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

from conv import conv_filter, conv_filter_batch

dataset_path = "dataset/"
image_size = (348, 348)

def load_image(image_path, label):
    file = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(file, channels=3)
    image = tf.image.resize(image, image_size)
    image = image / 255
    return image, label

def get_dataset(path, train, batch_size):
    df = pd.read_csv(dataset_path + path, header=None)
    df = df.drop(df.columns[-1], axis=1)
    df[0] = df[0].apply(lambda l: str(os.path.join(dataset_path, l)))
    image, labels = (df[0].values, df[1].values)
    x = tf.convert_to_tensor(image, dtype=tf.string)
    y = tf.convert_to_tensor(labels)
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(load_image)
    if train: dataset = dataset.shuffle(len(df))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def input_layer():
    model = Sequential()
    model.add(Input(shape=(348, 348, 3)))
    model.add(RandomFlip("horizontal"))
    model.add(RandomFlip("horizontal"))
    model.add(RandomRotation(0.2))
    model.add(RandomZoom(0.2))
    model.add(RandomContrast(0.2))
    model.add(RandomTranslation(0.1, 0.1))
    return model

def model1():
    model = Sequential()
    model.add(Input(shape=(348, 348, 3)))
    model.add(
        Conv2D(
            filters=8,
            kernel_size=3,
            strides=1,
            activation="relu"
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            filters=16,
            kernel_size=5,
            strides=1,
            activation="relu"
        )
    )
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(units=21, activation="relu"))
    # model.add(Dense(units=42, activation="relu"))
    model.add(Dense(units=81, activation="softmax"))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def model2():

    model = input_layer()

    conv_filter(model)
    model.add(Flatten())

    model.add(Dense(units=42, activation="relu"))
    model.add(Dense(units=81, activation="softmax"))
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def model3():

    model = input_layer()

    conv_filter_batch(model, size=[32, 64, 128])
    model.add(Flatten())

    model.add(Dense(units=512, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=256, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=81, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def model4():
    model = input_layer()

    conv_filter_batch(model, size=[64, 128, 256, 512])
    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=512, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=256, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(units=128, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units=81, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model