from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation


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
            kernel_size=3,
            strides=1,
            activation="relu"
        )
    )
    model.add(MaxPooling2D())
    model.add(
        Conv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            activation="relu"
        )
    )
    model.add(MaxPooling2D())
    model.add(Flatten())

def conv_filter_batch(model, size):

    for k in range(0, len(size)):
        model.add(
            Conv2D(
                filters=size[k],
                kernel_size=3,
                padding="same",
                strides=1,
            )
        )
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D())
