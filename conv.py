from keras.src.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten


def conv_filter(model):
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
            kernel_size=5,
            strides=1,
            activation="relu"
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            filters=128,
            kernel_size=5,
            strides=1,
            activation="relu"
        )
    )
    model.add(MaxPooling2D())
    model.add(Flatten())

def conv_filter_batch(model):
    model.add(
        Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
        )
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(
        Conv2D(
            filters=64,
            kernel_size=5,
            strides=1
        )
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(
        Conv2D(
            filters=128,
            kernel_size=5,
            strides=1
        )
    )
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())