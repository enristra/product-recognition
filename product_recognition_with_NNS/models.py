import tensorflow as tf
from keras_hub import models
from keras import Input, Model
from keras.layers import GlobalAveragePooling2D
from config import BATCH_SIZE, CHANNELS, IMG_HEIGHT, IMG_SIZE, IMG_WIDTH, PROTO_DIR, QUERY_DIR

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    # Media e deviazione standard ImageNet (BGR → qui usiamo RGB, keras_hub usa RGB)
    mean = tf.constant([0.485, 0.456, 0.406]) * 255.0
    std  = tf.constant([0.229, 0.224, 0.225]) * 255.0
    image = (image - mean) / std
    return image, label

def get_proto_dataset(shuffle=False):
    """
    Carica il dataset dei prototipi 
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        PROTO_DIR,
        image_size=IMG_SIZE,      
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        label_mode="int",          # label intere
    )
    # Salviamo i nomi delle classi prima di applicare map()
    class_names = dataset.class_names
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, class_names

def get_query_dataset(shuffle=False):
    """
    Carica il dataset delle query 
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        QUERY_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        label_mode="int",
    )
    class_names = dataset.class_names
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset, class_names

def resnet18_feature_extractor():
    """
    ResNet-18 pre-addestrata su ImageNet usata come feature extractor.
    Rimuoviamo la testa classificatoria e aggiungiamo un GlobalAveragePooling2D
    per ottenere un vettore di embedding 1-D per ogni immagine.
    """
    backbone = models.ResNetBackbone.from_preset("resnet_18_imagenet")
    backbone.trainable = False  # pesi congelati, siamo in inference mode

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    x = backbone(inputs) # output: feature map (7×7×512)
    x = GlobalAveragePooling2D()(x) # → vettore 512-D
    return Model(inputs, x, name="resnet18_feature_extractor")

def resnet50_feature_extractor():
    backbone = models.ResNetBackbone.from_preset("resnet_50_imagenet")
    backbone.trainable = False

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    x = backbone(inputs) # feature map (7×7×2048)
    x = GlobalAveragePooling2D()(x) # → vettore 2048-D
    return Model(inputs, x, name="resnet50_feature_extractor")

# Per DINOv2small e DINOv3small l’output della backbone è una sequenza di token, quindi si deve convertirla in un embedding 2D (N, D) prima della cosine similarity.
# Per ottenere un embedding globale dell’immagine è stato usato il CLS token 
def dinoV2small_feature_extractor():
    backbone = models.DINOV2Backbone.from_preset(
        "dinov2_small",
        image_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)) #resize dinamico dentro al modello da 518x518 a 224x224
    backbone.trainable = False

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)) 
    outputs = backbone({"images": inputs}) # feature map (16×16×384)
    x = outputs[:, 0, :] # CLS token -> shape: (batch, 384)
    return Model(inputs, x, name="dinoV2small_feature_extractor")

def dinoV3small_feature_extractor():
    backbone = models.DINOV3Backbone.from_preset("dinov3_vit_small_lvd1689m")
    backbone.trainable = False

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)) 
    outputs = backbone({"pixel_values": inputs}) # feature map (16×16×384)
    x = outputs[:, 0, :] # CLS token -> shape: (batch, 384)
    return Model(inputs, x, name="dinoV3small_feature_extractor")