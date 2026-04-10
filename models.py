import os.path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Input
from tensorflow.keras.layers import MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential

dataset_path = "dataset/"
image_size = (348, 348)

def load_image(image_path, label):
    file = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(file, channels=3)
    image = tf.image.resize(image, image_size)
    image = image / 255
    return image, label

def get_dataset(path, train):
    df = pd.read_csv(dataset_path + path, header=None)
    df = df.drop(df.columns[-1], axis=1)
    df[0] = df[0].apply(lambda l: str(os.path.join(dataset_path, l)))
    image, labels = (df[0].values, df[1].values)
    x = tf.convert_to_tensor(image, dtype=tf.string)
    y = tf.convert_to_tensor(labels)
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(load_image)
    if train: dataset = dataset.shuffle(len(df))
    dataset = dataset.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def model1():
    model = Sequential()
    model.add(Input(shape=(348, 348, 3)))
    model.add(
        Conv2D(
            filters=16,
            kernel_size=3,
            strides=1,
            activation="relu"
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            activation="relu"
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            activation="relu"
        )
    )
    model.add(MaxPooling2D(padding="same"))
    model.add(Flatten())
    model.add(Dense(units=42, activation="relu"))
    model.add(Dense(units=81, activation="softmax"))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model