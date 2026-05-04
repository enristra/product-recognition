import pandas as pd
import tensorflow as tf
import os
from keras import Model, Input
from keras.optimizers import Adam
from keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout,
    BatchNormalization, Activation,
    RandomFlip, RandomRotation, RandomZoom,
    RandomContrast, RandomTranslation
)
from keras.regularizers import l2
from keras_hub import models

dataset_path = "dataset/"

def load_image(image_path, label): # Carica, ridimensiona e preprocessa un'immagine per ResNet.
    file  = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(file, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image, label

def get_dataset(path, train, batch_size):
    df = pd.read_csv(os.path.join(dataset_path, path), header=None)
    df = df.drop(df.columns[-1], axis=1)
    df[0] = df[0].apply(lambda l: str(os.path.join(dataset_path, l)))

    x = tf.convert_to_tensor(df[0].values, dtype=tf.string)
    y = tf.convert_to_tensor(df[1].values)

    dataset = tf.data.Dataset.from_tensor_slices((x, y)).map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    if train:
        dataset = dataset.shuffle(len(df))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def model_resnet18_task2a(input_shape, dropout_rate, learning_rate, fine_tune=False):
    """
    Costruisce il modello base per il Task 2a:
    backbone ResNet-18 congelata + testa di classificazione identica al bestModel del Task 1.
    """

    backbone = models.ResNetBackbone.from_preset("resnet_18_imagenet")
    backbone.trainable = fine_tune # False: backbone congelato, True: fine-tuning

    inputs = Input(shape=input_shape)

    # Data augmentation — stessa del Task 1
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.2)(x)
    x = RandomZoom(0.2)(x)
    x = RandomContrast(0.2)(x)
    x = RandomTranslation(0.1, 0.1)(x)

    x = backbone(x, training=False)  # training=False: BN del backbone rimane in inferenza

    # Classification Head (identica al bestModel Task 1)
    x = GlobalAveragePooling2D()(x)

    wd = 1e-4  # L2 weight decay (regolarizzazione)
    x = Dense(units=128, use_bias=False, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(units=81, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="resnet18_task2a")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def model_resnet18_task2b(input_shape, dropout_rate, learning_rate):
    """
    Costruisce il modello per il Task 2b: fine-tuning della ResNet-18 con unfreeze parziale degli ultimi layer del backbone.
    """
    backbone = models.ResNetBackbone.from_preset("resnet_18_imagenet")
    
    # Sblocca tutto il backbone
    backbone.trainable = True
    # poi ricongela i primi layer (feature generiche ImageNet)
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    inputs = Input(shape=input_shape)

    # Data augmentation
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.2)(x)
    x = RandomZoom(0.2)(x)
    x = RandomContrast(0.2)(x)
    x = RandomTranslation(0.1, 0.1)(x)

    # training=True: ora la BN degli ultimi layer si adatta al nuovo dominio
    x = backbone(x, training=True)

    x = GlobalAveragePooling2D()(x)

    wd = 1e-4
    x = Dense(units=128, use_bias=False, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(units=81, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="resnet18_task2b")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model