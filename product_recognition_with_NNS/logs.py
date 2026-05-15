from pathlib import Path
import sys
from datetime import datetime


def logger(model, logs_dir):
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{model.name}_{timestamp}.txt"
    log_path = logs_dir / filename
    log_file  = open(log_path, "w", encoding="utf-8")

    original_stdout = sys.stdout

    class _Tee:
        def write(self, message):
            original_stdout.write(message)
            log_file.write(message)
            log_file.flush()
        def flush(self):
            original_stdout.flush()
            log_file.flush()

    sys.stdout = _Tee()
    print(f"Log salvato in: {log_path}")