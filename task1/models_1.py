import os.path
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import Dense, Input, BatchNormalization, Dropout, Activation, GlobalAveragePooling2D, Flatten, RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomTranslation, SpatialDropout2D
from keras import Sequential

from task1.conv_1 import conv_filter, conv_filter_batch, conv_filter_batchAndRegularizer
import task1.config_1 as config_1
from keras.regularizers import l2

dataset_path = "dataset/"

def load_image(image_path, label):
    file = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(file, channels=3)
    image = tf.image.resize(image, config_1.IMG_SIZE)
    image = image / 255
    return image, label

def get_dataset(path, train, batch_size):
    df = pd.read_csv(os.path.join(dataset_path, path), header=None)
    df = df.drop(df.columns[-1], axis=1)
    df[0] = df[0].apply(lambda l: str(os.path.join(dataset_path, l)))
    image, labels = (df[0].values, df[1].values)
    x = tf.convert_to_tensor(image, dtype=tf.string)
    y = tf.convert_to_tensor(labels)
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(load_image)
    if train: dataset = dataset.shuffle(len(df))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Definizione di input layer con data augmentation
def input_layer(model_name):
    model = Sequential(name=model_name)
    model.add(Input(shape=(config_1.IMG_HEIGHT, config_1.IMG_WIDTH, config_1.CHANNELS)))

    # Data augmentation
    model.add(RandomFlip("horizontal"))
    model.add(RandomRotation(0.2))
    model.add(RandomZoom(0.2))
    model.add(RandomContrast(0.2))
    model.add(RandomTranslation(0.1, 0.1))

    return model

# Model 1 - semplice CNN con 3 blocchi convoluzionali, un fully connected layer e un output layer
def model1():
    model = Sequential(name="model1")
    model.add(Input(shape=(config_1.IMG_HEIGHT, config_1.IMG_WIDTH, config_1.CHANNELS)))

    conv_filter(model)
    model.add(Flatten())

    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=81, activation="softmax"))
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Model 2 - CNN con batch normalization e data augmentation -> ridotto overfitting
def model2():
    model = input_layer("model2")

    conv_filter_batch(model, size=[32, 64, 128])
    model.add(Flatten())

    model.add(Dense(units=256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=81, activation="softmax"))
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Model 3 - CNN più profonda con global average pooling -> riduzione dei parametri e miglioramento della generalizzazione
def model3():
    model = input_layer("model3")

    conv_filter_batch(model, size=[32, 64, 128, 256])
    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.6))

    model.add(Dense(units=81, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=config_1.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Best model - aggiunta di regolarizzazione L2 per ridurre ulteriormente l'overfitting, miglioramento della generalizzazione e aumento dell'accuracy
def bestModel():
    model = input_layer("bestModel")

    wd = 1e-4
    conv_filter_batchAndRegularizer(model, size=[32, 64, 128, 256], wd=wd) 
    model.add(GlobalAveragePooling2D())

    model.add(Dense(units=128, use_bias=False, kernel_regularizer=l2(wd)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(config_1.DROPOUT_RATE))

    model.add(Dense(units=81, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=config_1.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ----------------------- Altri modelli -----------------------
# def model4():
#     model = input_layer("model4")

#     conv_filter_batch(model, size=[32, 64, 128, 256, 512])
#     model.add(GlobalAveragePooling2D())

#     model.add(Dense(units=512, use_bias=False))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(DROPOUT_RATE))

#     model.add(Dense(units=256, use_bias=False))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(DROPOUT_RATE))

#     model.add(Dense(units=128, use_bias=False))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(DROPOUT_RATE))

#     model.add(Dense(units=81, activation="softmax"))
#     model.compile(
#         optimizer=Adam(learning_rate=LEARNING_RATE),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model


# def model4withRegularizer():
#     model = input_layer("model4withRegularizer")

#     wd = 1e-4
#     conv_filter_batchAndRegularizer(model, size=[32, 64, 128, 256], wd=wd) #[32, 64, 128, 256, 512]
#     model.add(GlobalAveragePooling2D())

#     # model.add(Dense(units=512, use_bias=False, kernel_regularizer=l2(wd)))
#     # model.add(BatchNormalization())
#     # model.add(Activation('relu'))
#     # model.add(Dropout(DROPOUT_RATE))

#     # model.add(Dense(units=256, use_bias=False, kernel_regularizer=l2(wd)))
#     # model.add(BatchNormalization())
#     # model.add(Activation('relu'))
#     # model.add(Dropout(DROPOUT_RATE))

#     model.add(Dense(units=128, use_bias=False, kernel_regularizer=l2(wd)))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(DROPOUT_RATE))

#     model.add(Dense(units=81, activation="softmax"))
#     model.compile(
#         optimizer=Adam(learning_rate=LEARNING_RATE),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model


# def model5withRegularizer(): #model con regolarizzazione più leggera rispetto a model5, riesce ad imparare meglio ma overfitta
#     model = input_layer("model5withRegularizer")

#     wd = 1e-4
#     conv_filter_batchAndRegularizer(model, size=[32, 64, 128], wd=wd)

#     # Blocco 4
#     model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=l2(wd))),
#     model.add(BatchNormalization()), 
#     model.add(Activation('relu')),
#     model.add(MaxPooling2D(2,2)),
#     model.add(SpatialDropout2D(0.05)),
    
#     model.add(GlobalAveragePooling2D())

#     model.add(Dense(units=256, use_bias=False, kernel_regularizer=l2(wd)))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(DROPOUT_RATE))

#     model.add(Dense(units=81, activation="softmax"))
#     model.compile(
#         optimizer=Adam(learning_rate=LEARNING_RATE),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model


# def model5(): #troppo regolarizzato, non riesce ad imparare
#     model = input_layer("model5")

#     conv_filter_batch(model, size=[32, 64])

#     # Blocco 3
#     model.add(Conv2D(128, (3,3), padding='same')),
#     model.add(BatchNormalization()), 
#     model.add(Activation('relu')),
#     model.add(MaxPooling2D(2,2)),
#     model.add(SpatialDropout2D(0.1)), 

#     # Blocco 4
#     model.add(Conv2D(256, (3,3), padding='same')),
#     model.add(BatchNormalization()), 
#     model.add(Activation('relu')),
#     model.add(MaxPooling2D(2,2)),
#     model.add(SpatialDropout2D(0.15)),
    
#     model.add(GlobalAveragePooling2D())

#     model.add(Dense(units=256, use_bias=False))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(DROPOUT_RATE))

#     model.add(Dense(units=81, activation="softmax"))
#     model.compile(
#         optimizer=Adam(learning_rate=LEARNING_RATE),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model