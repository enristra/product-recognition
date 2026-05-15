import keras
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from logs import save_experiment_log
from plotting import plot
from task2.config_2 import SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, DROPOUT_RATE, IMG_HEIGHT, IMG_WIDTH, CHANNELS
from task2.models_2 import model_resnet18_task2a, get_dataset, model_resnet18_task2b

def set_seed(seed):
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

def main():

    set_seed(SEED)

    train_dataset = get_dataset("train.txt", train=True,  batch_size=BATCH_SIZE)
    val_dataset   = get_dataset("val.txt",   train=False, batch_size=BATCH_SIZE)
    test_dataset  = get_dataset("test.txt",  train=False, batch_size=BATCH_SIZE)

    model = model_resnet18_task2b((IMG_HEIGHT, IMG_WIDTH, CHANNELS), DROPOUT_RATE, LEARNING_RATE)
    print(model.summary())

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=40,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            verbose=1,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Valutazione sul test set
    test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
    
    plot(history, model.name)
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