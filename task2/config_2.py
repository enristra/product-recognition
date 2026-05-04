# Parametri input
IMG_HEIGHT = 224   # ResNet richiede almeno 224x224
IMG_WIDTH  = 224
IMG_SIZE   = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS   = 3

# Training
BATCH_SIZE    = 32
EPOCHS        = 200
LEARNING_RATE = 0.001 
SEED          = 42

# Modello
DROPOUT_RATE  = 0.6  