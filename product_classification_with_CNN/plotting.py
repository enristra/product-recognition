from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

def plot_validation(history, label):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(label + ' Accuracy')
    plt.legend()
    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

def plot(history, model_name):
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Training vs Validation Accuracy")
    # plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Salvataggio
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{model_name}_{timestamp}_plot.png"
    plt.savefig(plots_dir / filename)

    plt.show()

def plot_multitask(history, model_name):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Fine class accuracy
    axes[0, 0].plot(history.history['f_accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_f_accuracy'], label='Val')
    axes[0, 0].set_title("Fine Class (81) - Accuracy")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Fine class loss
    axes[1, 0].plot(history.history['f_loss'], label='Train')
    axes[1, 0].plot(history.history['val_f_loss'], label='Val')
    axes[1, 0].set_title("Fine Class (81) - Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Super class accuracy
    axes[0, 1].plot(history.history['s_accuracy'], label='Train')
    axes[0, 1].plot(history.history['val_s_accuracy'], label='Val')
    axes[0, 1].set_title("Super Class (43) - Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Super class loss
    axes[1, 1].plot(history.history['s_loss'], label='Train')
    axes[1, 1].plot(history.history['val_s_loss'], label='Val')
    axes[1, 1].set_title("Super Class (43) - Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.suptitle(f"{model_name} - Multi-Task Training", fontsize=14)
    plt.tight_layout()

    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    plt.savefig(plots_dir / f"{model_name}_{timestamp}_plot.png")
    plt.show()