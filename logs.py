from datetime import datetime
from pathlib import Path

def save_experiment_log(
    model,
    history,
    test_loss,
    test_acc,
    config_dict,
    logs_dir,
):
    """
    Salva un report testuale dell'esperimento
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{model.name}_{timestamp}.txt"
    log_path = logs_dir / filename

    hist = history.history

    train_acc = hist.get("accuracy", [])
    val_acc = hist.get("val_accuracy", [])
    train_loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])

    best_epoch = max(range(len(val_acc)), key=lambda i: val_acc[i]) + 1 if val_acc else None
    best_train_acc = max(train_acc) if train_acc else None
    best_val_acc = max(val_acc) if val_acc else None
    best_train_loss = min(train_loss) if train_loss else None
    best_val_loss = min(val_loss) if val_loss else None

    final_train_acc = train_acc[-1] if train_acc else None
    final_val_acc = val_acc[-1] if val_acc else None
    final_train_loss = train_loss[-1] if train_loss else None
    final_val_loss = val_loss[-1] if val_loss else None

    # Cattura model.summary() in una stringa
    model_summary_lines = []
    model.summary(print_fn=lambda x: model_summary_lines.append(x))
    model_summary_text = "\n".join(model_summary_lines)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"EXPERIMENT REPORT {filename}\n")
        f.write("=" * 60 + "\n")

        f.write(f"Timestamp: {timestamp}\n\n")

        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 60 + "\n")
        f.write(model_summary_text)
        f.write("\n")

        f.write("CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("BEST METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Best epoch (based on val_accuracy): {best_epoch}\n")
        f.write(f"Best train accuracy: {best_train_acc:.4f}\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"Best train loss: {best_train_loss:.4f}\n")
        f.write(f"Best validation loss: {best_val_loss:.4f}\n\n")

        f.write("FINAL EPOCH METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Final train accuracy: {final_train_acc:.4f}\n")
        f.write(f"Final validation accuracy: {final_val_acc:.4f}\n")
        f.write(f"Final train loss: {final_train_loss:.4f}\n")
        f.write(f"Final validation loss: {final_val_loss:.4f}\n\n")

        f.write("TEST METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Test accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Test loss: {test_loss:.4f}")
        
    print(f"\nLog esperimento salvato in: {log_path}")