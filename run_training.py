#!/usr/bin/env python3
"""Launch the LeNet-5 MNIST training script."""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_DIR / "python" / "train_lenet.py"
VENV_PYTHON = PROJECT_DIR / ".venv" / "bin" / "python"


def main():
    if not TRAIN_SCRIPT.is_file():
        sys.exit(f"ERROR: Training script not found:\n{TRAIN_SCRIPT}")

    if not VENV_PYTHON.is_file():
        sys.exit(
            f"ERROR: Virtual environment not found:\n{VENV_PYTHON}\n"
            "Create it with: python3 -m venv .venv"
        )

    epochs = input("Number of training epochs [8]: ").strip() or "8"

    if not epochs.isdigit() or int(epochs) < 1:
        sys.exit("ERROR: Enter a positive integer for epochs.")

    command = [
        str(VENV_PYTHON),
        str(TRAIN_SCRIPT),
        "--epochs",
        epochs,
    ]

    print("\nStarting LeNet-5-style MNIST training...")
    print(f"Project: {PROJECT_DIR}")
    print(f"Epochs:  {epochs}")
    print("Device:  CPU\n")

    try:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(130)
    except subprocess.CalledProcessError as error:
        sys.exit(f"Training failed with exit code {error.returncode}.")

    print("\nTraining finished. Check models/, logs/, and outputs/.")


if __name__ == "__main__":
    main()
