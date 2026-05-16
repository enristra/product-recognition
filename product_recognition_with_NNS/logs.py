from pathlib import Path
import sys
from datetime import datetime

def logger(model, logs_dir):
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{model.name}_{timestamp}.txt"
    log_path = logs_dir / filename

    # Chiude il log precedente se stdout è già un _Tee
    if hasattr(sys.stdout, '_log_file'):
        sys.stdout._log_file.close()
        sys.stdout = sys.stdout._original_stdout  # ripristina stdout originale

    log_file  = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout

    class _Tee:
        def __init__(self):
            self._log_file = log_file
            self._original_stdout = original_stdout

        def write(self, message):
            original_stdout.write(message)
            log_file.write(message)
            log_file.flush()

        def flush(self):
            original_stdout.flush()
            log_file.flush()

    sys.stdout = _Tee()
    print(f"Log salvato in: {log_path}")