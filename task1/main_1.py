import argparse
import keras
import numpy as np
import task1.config_1 as config_1
from logs import save_experiment_log
import task1.models_1 as models
from plotting import plot
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def set_seed(seed):
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser("Training CNN - task1")
    parser.add_argument("--model", type=str, required=True,
                        choices=config_1.MODEL_REGISTRY.keys(), 
                        help=f"Nome del modello da allenare. Scelte disponibili: {list(config_1.MODEL_REGISTRY.keys())}")
    return parser.parse_args()

def main():

    set_seed(config_1.SEED)

    train_dataset = models.get_dataset("train.txt", True, config_1.BATCH_SIZE)
    val_dataset = models.get_dataset("val.txt", False, config_1.BATCH_SIZE)
    test_dataset = models.get_dataset("test.txt", False, config_1.BATCH_SIZE)

    # Scelta del modello da addestrare da terminale
    args = parse_args()
    name_model = args.model
    model = config_1.MODEL_REGISTRY[name_model]()
    print(model.summary())

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=config_1.EPOCHS, callbacks=[
        EarlyStopping( # Early stopping per evitare overfitting: se la validation accuracy non migliora per 6 epoche, fermo il training
            monitor="val_accuracy",
            patience=40, 
            mode="max",
            restore_best_weights=True,
            verbose=1,),
        ReduceLROnPlateau( # Riduzione del learning rate quando la validation loss non migliora -> fine tuning automatico
            monitor="val_loss",
            factor=0.5,
            patience=8, 
            verbose=1,
            min_lr=1e-5,
        ),
    ])

    test = model.evaluate(test_dataset)
    test_loss, test_acc = test[0], test[1]

    plot(history, model.name)

    print(f"Test Accuracy: {test_acc * 100:.2f}%\nTest Loss: {test_loss}")

    # Salvataggio del log
    config_log = {
        "img_height": config_1.IMG_HEIGHT,
        "img_width": config_1.IMG_WIDTH,
        "channels": config_1.CHANNELS,
        "batch_size": config_1.BATCH_SIZE,
        "epochs": config_1.EPOCHS,
        "learning_rate": config_1.LEARNING_RATE,
        "dropout_rate": config_1.DROPOUT_RATE,
        "seed": config_1.SEED,
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