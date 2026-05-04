from task1 import models_1 as models

# Parametri input
IMG_HEIGHT = 192 #348
IMG_WIDTH = 192 #348
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3

# Training
BATCH_SIZE = 32 #8
EPOCHS = 200 #100
LEARNING_RATE = 0.001 #0.0005
SEED = 42

# Modello
DROPOUT_RATE = 0.6

MODEL_REGISTRY = {
    "model1": models.model1,
    "model2": models.model2,
    "model3": models.model3,
    "bestModel": models.bestModel,
    # "model4": models.model4,
    # "model4withRegularizer": models.model4withRegularizer,
    # "model5": models.model5,
    # "model5withRegularizer": models.model5withRegularizer,
}