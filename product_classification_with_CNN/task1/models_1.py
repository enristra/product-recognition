import os.path
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam
from keras.layers import Concatenate, Conv2D, Dense, Input, BatchNormalization, Dropout, Activation, GlobalAveragePooling2D, Flatten, MaxPooling2D, RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomTranslation, SpatialDropout2D
from keras import Model
from keras.regularizers import l2

from task1.config_1 import CHANNELS, DROPOUT_RATE, IMG_HEIGHT, IMG_SIZE, IMG_WIDTH, LEARNING_RATE

dataset_path = "dataset/"

def load_image(image_path, label):
    file = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(file, channels=CHANNELS)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255
    return image, label

def get_dataset(path, train, batch_size, super_label=False):
    df = pd.read_csv(os.path.join(dataset_path, path), header=None)
    df[0] = df[0].apply(lambda l: str(os.path.join(dataset_path, l.strip())))

    image = df[0].values
    x = tf.convert_to_tensor(image, dtype=tf.string)
    y_fine = tf.convert_to_tensor(df[1].astype(int).values, dtype=tf.int32)

    if super_label:
        y_coarse = tf.convert_to_tensor(df[2].astype(int).values, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((x, {'s': y_coarse, 'f': y_fine}))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x, y_fine))

    dataset = dataset.map(load_image)
    if train: dataset = dataset.shuffle(len(df))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# def get_dataset(path, train, batch_size):
#     df = pd.read_csv(os.path.join(dataset_path, path), header=None)
#     df = df.drop(df.columns[-1], axis=1)
#     df[0] = df[0].apply(lambda l: str(os.path.join(dataset_path, l)))
#     image, labels = (df[0].values, df[1].values)
#     x = tf.convert_to_tensor(image, dtype=tf.string)
#     y = tf.convert_to_tensor(labels)
#     dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(load_image)
#     if train: dataset = dataset.shuffle(len(df))
#     dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
#     return dataset

# Definizione di input layer 
def input_layer(data_augmentation = True):
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    x = inputs   # variabile separata per costruire il grafo

    if data_augmentation:
        x = RandomFlip("horizontal")(x)
        x = RandomRotation(0.2)(x)
        x = RandomZoom(0.2)(x)
        x = RandomContrast(0.2)(x)
        x = RandomTranslation(0.1, 0.1)(x)

    return inputs, x 

# Model 1 - semplice CNN con 3 blocchi convoluzionali, un fully connected layer e un output layer
def model1():
    inputs, x = input_layer(False) # senza data augmentation

    x = Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)

    x = Dense(units=256, activation="relu")(x)

    outputs = Dense(units=81, activation="softmax")(x)
    
    model = Model(name="model1", inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Model 2 - CNN con batch normalization e data augmentation -> ridotto overfitting
def model2():
    inputs, x = input_layer()

    x = Conv2D(filters=16, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=32, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=128, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)

    x = Dense(units=162, use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=162, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(DROPOUT_RATE)(x)

    outputs = Dense(units=81, activation="softmax")(x)

    model = Model(name="model2", inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=False #?
    )
    return model

# Model 3 - CNN più profonda con global average pooling -> riduzione dei parametri e miglioramento della generalizzazione
def model3():
    inputs, x = input_layer()

    x = Conv2D(filters=32, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=128, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=256, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(units=128, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(DROPOUT_RATE)(x)

    outputs = Dense(units=81, activation="softmax")(x)

    model = Model(name="model3", inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Model 4 - Best model: aggiunta di regolarizzazione L2 per ridurre ulteriormente l'overfitting, miglioramento della generalizzazione e aumento dell'accuracy
def model4():
    wd = 1e-4
    inputs, x = input_layer()

    x = Conv2D(filters=32, kernel_size=3, padding="same", kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=64, kernel_size=3, padding="same", kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=128, kernel_size=3, padding="same", kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=256, kernel_size=3, padding="same", kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(units=128, use_bias=False, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(DROPOUT_RATE)(x)

    outputs = Dense(units=81, activation="softmax")(x)

    model = Model(name="model4", inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Model 5 - CNN con multi-task learning per classificazione gerarchica -> miglioramento dell'accuracy e della generalizzazione grazie alla condivisione delle rappresentazioni tra i due task
def model5():
    wd = 1e-3
    inputs, x = input_layer()

    conv = Conv2D(filters=16, kernel_size=3, kernel_regularizer=l2(wd))(x)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = MaxPooling2D()(conv)

    conv = Conv2D(filters=32, kernel_size=3, kernel_regularizer=l2(wd))(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = MaxPooling2D()(conv)

    conv = Conv2D(filters=64, kernel_size=3, kernel_regularizer=l2(wd))(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = MaxPooling2D()(conv)

    conv = Conv2D(filters=128, kernel_size=3, kernel_regularizer=l2(wd))(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = MaxPooling2D()(conv)

    conv = Conv2D(filters=256, kernel_size=3, kernel_regularizer=l2(wd))(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = MaxPooling2D()(conv)

    # conv = Conv2D(filters=256, kernel_size=3, kernel_regularizer=l2(wd))(conv)
    # conv = BatchNormalization()(conv)
    # conv = Activation('relu')(conv)
    # conv = MaxPooling2D()(conv)

    # conv = Dropout(DROPOUT_RATE)(conv)

    conv = GlobalAveragePooling2D()(conv)

    # dense = Dense(units=324, use_bias=False, kernel_regularizer=l2(wd))(conv)
    # dense = Dropout(DROPOUT_RATE)(dense)
    # dense = Activation('relu')(dense)
    # dense = BatchNormalization()(dense)

    dense = Dense(units=128, use_bias=False, kernel_regularizer=l2(wd))(conv)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dropout(DROPOUT_RATE)(dense)

    super_class = Dense(units=43, activation='softmax', name='s')(dense)
    combined = Concatenate()([dense, super_class])
    fine_class = Dense(units=81, activation='softmax', name='f')(combined)
    model = Model(name = "model5", inputs=inputs, outputs=[super_class, fine_class])
    model.compile(
        optimizer=Adam(LEARNING_RATE),
        jit_compile=False, #?
        loss={
            's': 'sparse_categorical_crossentropy',
            'f': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            's': 0.8,
            'f': 1.0
        },
        metrics={
            's': 'accuracy',
            'f': 'accuracy',
        }
    )
    return model

# ----------------------- Altri modelli -----------------------

# # MineModel 2 - CNN con batch normalization e data augmentation -> ridotto overfitting
# def MineModel2():
#     model = input_layer("MineModel2")

#     conv_filter_batch(model, size=[32, 64, 128])
#     model.add(Flatten())

#     model.add(Dense(units=256, activation="relu"))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))

#     model.add(Dense(units=81, activation="softmax"))
#     model.compile(
#         optimizer=Adam(),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model

# def model5withRegularizer(): #model con regolarizzazione più leggera rispetto a model5, riesce ad imparare meglio ma overfitta
#     model = input_layer("model5withRegularizer")

#     wd = 1e-4

#     model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(wd))),
#     model.add(BatchNormalization()), 
#     model.add(Activation('relu')),
#     model.add(MaxPooling2D(2,2)),

#     model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=l2(wd))),
#     model.add(BatchNormalization()), 
#     model.add(Activation('relu')),
#     model.add(MaxPooling2D(2,2)),

#     model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=l2(wd))),
#     model.add(BatchNormalization()), 
#     model.add(Activation('relu')),
#     model.add(MaxPooling2D(2,2)),

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
