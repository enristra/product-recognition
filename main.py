import keras
import numpy as np
from logs import save_experiment_log
import models
from plotting import plot_validation, plot_loss, plot
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Parametri input
IMG_HEIGHT = 192 #348
IMG_WIDTH = 192 #348
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3

# Training
BATCH_SIZE = 16 #8
EPOCHS = 50 #100
LEARNING_RATE = 0.001
SEED = 42

# Modello
DROPOUT_RATE = 0.5

def set_seed(seed):
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def main():

    # set_seed(SEED)

    train_dataset = models.get_dataset("train.txt", True, BATCH_SIZE)
    val_dataset = models.get_dataset("val.txt", False, BATCH_SIZE)
    test_dataset = models.get_dataset("test.txt", False, BATCH_SIZE)

    # model1 = models.model1()
    # print(model1.summary())
    #
    # history2 = model1.fit(train_dataset, validation_data=val_dataset, epochs=10)
    # test = model1.evaluate(test_dataset)
    # plot_validation(history2, 'Validation')
    # plot_loss(history2)
    # print(f"Test Accuracy: {test[1] * 100:.2f}%\nTest Loss: {test[0]}")

    model = models.model4withRegularizer()
    print(model.summary())

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[
        # EarlyStopping( # Early stopping per evitare overfitting: se la validation accuracy non migliora per 6 epoche, fermo il training
        #     monitor="val_accuracy",
        #     patience=8, #6
        #     mode="max",
        #     restore_best_weights=True,
        #     verbose=1,),
        ReduceLROnPlateau( # Riduzione del learning rate quando la validation loss non migliora -> fine tuning automatico
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6,
        ),
    ])

    test = model.evaluate(test_dataset)
    test_loss, test_acc = test[0], test[1]

    plot(history, model.name)
    # plot_validation(history2, 'Validation')
    # plot_loss(history2)
    print(f"Test Accuracy: {test_acc * 100:.2f}%\nTest Loss: {test_loss}")

    # Salvataggio del log
    config_log = {
        "img_height": IMG_HEIGHT,
        "img_width": IMG_WIDTH,
        "channels": CHANNELS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "dropout_rate": DROPOUT_RATE,
        "seed": SEED,
    }

    save_experiment_log(
        model=model,
        history=history,
        test_loss=test_loss,
        test_acc=test_acc,
        config_dict=config_log,
        logs_dir="outputs/logs",
    )

if __name__ == "__main__":
    main()