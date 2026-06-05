import argparse
import keras
import numpy as np
from task1.config_1 import BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED, DROPOUT_RATE, IMG_HEIGHT, IMG_WIDTH, CHANNELS
from logs import save_log, save_multitask_log
import task1.models_1 as models
from plotting import plot, plot_multitask
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

MODEL_REGISTRY = {
    "model1": models.model1,
    "model2": models.model2,
    "model3": models.model3,
    "model4": models.model4,
    "model5": models.model5,
}

def set_seed(seed):
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser("Training CNN - task1")
    parser.add_argument("--model", type=str, required=True,
                        choices=MODEL_REGISTRY.keys(), 
                        help=f"Nome del modello da allenare. Scelte disponibili: {list(MODEL_REGISTRY.keys())}")
    return parser.parse_args()

def main():

    set_seed(SEED)

    # Scelta del modello da addestrare da terminale
    args = parse_args()
    name_model = args.model
    is_multitask = (name_model == "model5")

    train_dataset = models.get_dataset("train.txt", True, BATCH_SIZE, is_multitask)
    val_dataset = models.get_dataset("val.txt", False, BATCH_SIZE, is_multitask)
    test_dataset = models.get_dataset("test.txt", False, BATCH_SIZE, is_multitask)

    model = MODEL_REGISTRY[name_model]()
    print(model.summary())

    monitor_acc  = "val_f_accuracy" if is_multitask else "val_accuracy"
    monitor_loss = "val_f_loss"     if is_multitask else "val_loss"

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[
        EarlyStopping(
            monitor=monitor_acc,
            patience=40,
            mode="max",
            restore_best_weights=True,
            verbose=1,),
        ReduceLROnPlateau(
            monitor=monitor_loss,
            factor=0.5,
            patience=8,
            verbose=1,
            min_lr=1e-5,
        ),
    ])

    test = model.evaluate(test_dataset, return_dict=True)

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

    if name_model == "model5":
        plot_multitask(history, model.name)
        save_multitask_log(model, history, test, config_log, "outputs/logs")
    else:
        plot(history, model.name)
        save_log(model, history, test, config_log, "outputs/logs")

if __name__ == "__main__":
    main()