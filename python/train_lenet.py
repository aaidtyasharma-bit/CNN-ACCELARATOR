#!/usr/bin/env python3
"""
LeNet-5 Training & Hardware Export Script for MNIST.
Optimized for Hardware Accelerator Development with:
  - Data Augmentation (RandomAffine: translation, slight rotation) for robustness.
  - Learning Rate Scheduler (CosineAnnealingLR) for smooth convergence.
  - Dropout regularization (fc1) to prevent overfitting while preserving zero-cost inference in hardware.
  - Automatic export of INT8 weights, Verilog HEX files, C headers, and test stimulus.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LeNet5(nn.Module):
    """
    Classic LeNet-5 architecture with optional Dropout (training only).
    During hardware inference / evaluation, dropout is inactive (eval mode).
      - Conv1: 1 -> 6 (5x5, padding=2) => 28x28
      - Pool1: 2x2 MaxPool (stride=2)   => 14x14
      - Conv2: 6 -> 16 (5x5, padding=0) => 10x10
      - Pool2: 2x2 MaxPool (stride=2)   => 5x5
      - Flatten: 16 * 5 * 5 = 400
      - FC1: 400 -> 120 (ReLU)
      - Dropout (p=0.25, training only)
      - FC2: 120 -> 84 (ReLU)
      - FC3: 84 -> 10 (Logits)
    """

    def __init__(self, dropout_rate: float = 0.25):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


def quantize_to_int8(tensor_np: np.ndarray):
    """Symmetric per-tensor INT8 quantization."""
    max_val = np.max(np.abs(tensor_np))
    scale = float(max_val / 127.0) if max_val > 0 else 1.0
    q_int = np.clip(np.round(tensor_np / scale), -128, 127).astype(np.int8)
    return q_int, scale


def to_hex2(val_int8: int) -> str:
    """Format signed 8-bit int into 2-digit uppercase hex (two's complement)."""
    return f"{(int(val_int8) & 0xFF):02X}"


def export_hardware_artifacts(model, hw_dir: Path):
    """
    Generate hardware-ready files:
      - .hex files for each layer for Verilog $readmemh.
      - Monolithic memory image (lenet5_all_weights.hex).
      - C header (lenet5_weights.h).
      - Hardware memory descriptor (hardware_descriptor.json).
    """
    hw_dir.mkdir(parents=True, exist_ok=True)
    hex_dir = hw_dir / "hex"
    hex_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "architecture": "LeNet-5",
        "quantization": "Symmetric INT8 (-128 to 127)",
        "layers": {},
        "total_parameters": 0,
        "total_weight_bytes": 0,
    }

    c_header_lines = [
        "/* Auto-generated LeNet-5 INT8 Quantized Parameters for Hardware Accelerator */",
        "#ifndef LENET5_WEIGHTS_H",
        "#define LENET5_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "",
    ]

    combined_hex_lines = []
    current_offset = 0

    for name, param in model.named_parameters():
        clean_name = name.replace(".", "_")
        arr = param.detach().cpu().numpy()
        q_arr, scale = quantize_to_int8(arr)
        flat = q_arr.flatten()

        num_elements = len(flat)
        start_addr = current_offset
        end_addr = current_offset + num_elements - 1
        current_offset += num_elements

        # 1. Per-layer HEX
        layer_hex_path = hex_dir / f"{clean_name}.hex"
        with open(layer_hex_path, "w") as f:
            for val in flat:
                f.write(f"{to_hex2(val)}\n")

        # 2. Combined HEX lines
        for val in flat:
            combined_hex_lines.append(f"{to_hex2(val)}\n")

        # 3. C Header
        c_header_lines.append(f"// Layer: {name} | Scale: {scale:.8e} | Shape: {list(arr.shape)}")
        c_header_lines.append(f"static const float {clean_name}_scale = {scale:.8e}f;")
        c_header_lines.append(f"static const int8_t {clean_name}[{num_elements}] = {{")
        chunk_size = 16
        for i in range(0, num_elements, chunk_size):
            chunk = flat[i:i + chunk_size]
            c_header_lines.append("    " + ", ".join(f"{int(v)}" for v in chunk) + ",")
        c_header_lines.append("};\n")

        metadata["layers"][name] = {
            "c_name": clean_name,
            "shape": list(arr.shape),
            "num_elements": num_elements,
            "scale": scale,
            "bram_start_addr": start_addr,
            "bram_end_addr": end_addr,
            "hex_file": f"hex/{clean_name}.hex",
        }
        metadata["total_parameters"] += num_elements

    metadata["total_weight_bytes"] = current_offset
    c_header_lines.append("#endif // LENET5_WEIGHTS_H\n")

    # Write combined hex
    combined_hex_path = hw_dir / "lenet5_all_weights.hex"
    with open(combined_hex_path, "w") as f:
        f.writelines(combined_hex_lines)

    # Write C header
    c_header_path = hw_dir / "lenet5_weights.h"
    with open(c_header_path, "w") as f:
        f.write("\n".join(c_header_lines))

    # Write JSON descriptor
    json_path = hw_dir / "hardware_descriptor.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def export_sample_test_inputs(test_loader, hw_dir: Path, num_samples: int = 10):
    """Export test vectors and golden outputs for Verilog testbenches."""
    tb_dir = hw_dir / "test_stimulus"
    tb_dir.mkdir(parents=True, exist_ok=True)

    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    stimulus_info = []

    for i in range(min(num_samples, len(images))):
        img_np = images[i].squeeze().numpy()
        label = int(labels[i].item())

        q_img, img_scale = quantize_to_int8(img_np)
        flat_img = q_img.flatten()

        hex_file = tb_dir / f"test_img_{i}_digit_{label}.hex"
        with open(hex_file, "w") as f:
            for val in flat_img:
                f.write(f"{to_hex2(val)}\n")

        stimulus_info.append({
            "sample_id": i,
            "true_label": label,
            "hex_file": f"test_stimulus/{hex_file.name}",
            "scale": img_scale,
            "pixels": len(flat_img)
        })

    with open(tb_dir / "stimulus_manifest.json", "w") as f:
        json.dump(stimulus_info, f, indent=2)


def plot_metrics(history, output_path: Path):
    """Plot training and validation loss, accuracy, and learning rate curves."""
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs_range, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs_range, history["test_loss"], "r--s", label="Test Loss")
    ax1.set_title("Cross-Entropy Loss vs Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_xticks(list(epochs_range))

    ax2.plot(epochs_range, history["train_acc"], "b-o", label="Train Accuracy")
    ax2.plot(epochs_range, history["test_acc"], "g--^", label="Test Accuracy")
    ax2.set_title("Accuracy (%) vs Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_xticks(list(epochs_range))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_sample_predictions(model, test_loader, device, output_path: Path):
    """Generate visual grid of sample predictions for verification."""
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze().numpy()
        true_label = labels[i].item()
        pred_label = preds[i].item()
        color = "green" if true_label == pred_label else "red"

        ax.imshow(img, cmap="gray")
        ax.set_title(f"True: {true_label} | Pred: {pred_label}", color=color, fontsize=10)
        ax.axis("off")

    plt.suptitle("LeNet-5 MNIST Sample Predictions", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train LeNet-5 with Enhanced Hyperparameters & Export for Hardware")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (L2 regularization)")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout probability for FC layer")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Path to models directory")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Path to logs directory")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Path to outputs directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = (project_root / args.data_dir).resolve()
    models_dir = (project_root / args.models_dir).resolve()
    hw_dir = models_dir / "hardware"
    logs_dir = (project_root / args.logs_dir).resolve()
    outputs_dir = (project_root / args.outputs_dir).resolve()

    for d in [data_dir, models_dir, hw_dir, logs_dir, outputs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "training_log.txt"
    log_fp = open(log_file, "w", encoding="utf-8")

    def log_print(msg: str):
        print(msg)
        log_fp.write(msg + "\n")
        log_fp.flush()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    log_print("=" * 70)
    log_print("  LeNet-5 Training & Hardware Export (Optimized Hyperparameters)  ")
    log_print("=" * 70)
    log_print(f"Project Directory:  {project_root}")
    log_print(f"Device:             {device}")
    log_print(f"Epochs:             {args.epochs}")
    log_print(f"Batch Size:         {args.batch_size}")
    log_print(f"Initial LR:         {args.lr}")
    log_print(f"Weight Decay:       {args.weight_decay}")
    log_print(f"Dropout Rate:       {args.dropout}")
    log_print(f"LR Scheduler:       CosineAnnealingLR (T_max={args.epochs})")
    log_print(f"Data Augmentation:  RandomAffine(degrees=10, translate=(0.06, 0.06))")
    log_print("-" * 70)

    # 1. Enhanced Data Transforms: Data Augmentation for training, pure normalization for testing
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.06, 0.06), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    log_print("Loading MNIST dataset with augmentation...")
    train_dataset = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

    log_print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    log_print("-" * 70)

    # 2. Model, Optimizer, Scheduler
    model = LeNet5(dropout_rate=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
    }

    best_test_acc = 0.0
    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        epoch_train_loss = running_loss / total
        epoch_train_acc = (correct / total) * 100.0

        # Evaluation (eval mode disables dropout)
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        epoch_test_loss = test_loss / test_total
        epoch_test_acc = (test_correct / test_total) * 100.0
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["test_loss"].append(epoch_test_loss)
        history["test_acc"].append(epoch_test_acc)
        history["lr"].append(current_lr)

        log_print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {epoch_train_loss:.4f} (Acc: {epoch_train_acc:5.2f}%) | "
            f"Test Loss: {epoch_test_loss:.4f} (Acc: {epoch_test_acc:5.2f}%) | "
            f"Time: {epoch_time:.1f}s"
        )

        if epoch_test_acc > best_test_acc:
            best_test_acc = epoch_test_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": best_test_acc,
                "arch": "LeNet-5",
                "hyperparameters": {
                    "dropout": args.dropout,
                    "weight_decay": args.weight_decay,
                    "lr": args.lr,
                    "scheduler": "CosineAnnealingLR",
                    "augmentation": "RandomAffine",
                }
            }, models_dir / "lenet5_best.pth")

    total_time = time.time() - total_start_time
    log_print("-" * 70)
    log_print(f"Training completed in {total_time:.1f}s")
    log_print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    log_print("-" * 70)

    # Save final model state
    final_pth = models_dir / "lenet5_final.pth"
    torch.save(model.state_dict(), final_pth)
    log_print(f"Saved PyTorch weights: {final_pth}")

    # Generate Hardware Artifacts (INT8, HEX, C header)
    log_print("\nExporting Hardware INT8 Artifacts...")
    hw_meta = export_hardware_artifacts(model, hw_dir)
    log_print(f"  - Verilog Memory HEX (.mem/.hex): {hw_dir / 'hex'}")
    log_print(f"  - Combined Memory Image:          {hw_dir / 'lenet5_all_weights.hex'} ({hw_meta['total_weight_bytes']} bytes)")
    log_print(f"  - C Header File:                  {hw_dir / 'lenet5_weights.h'}")
    log_print(f"  - Hardware Descriptor:            {hw_dir / 'hardware_descriptor.json'}")

    export_sample_test_inputs(test_loader, hw_dir, num_samples=10)
    log_print(f"  - Verilog Test Stimulus:          {hw_dir / 'test_stimulus'}")

    # Save curves & predictions
    plot_path = outputs_dir / "loss_accuracy_curve.png"
    plot_metrics(history, plot_path)
    log_print(f"\nSaved training curves: {plot_path}")

    pred_path = outputs_dir / "sample_predictions.png"
    save_sample_predictions(model, test_loader, device, pred_path)
    log_print(f"Saved sample predictions: {pred_path}")

    summary_path = logs_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "architecture": "LeNet-5",
            "epochs": args.epochs,
            "best_test_accuracy": best_test_acc,
            "total_time_seconds": total_time,
            "hyperparameters": {
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "learning_rate": args.lr,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "augmentation": "RandomAffine(10deg, translate=0.06)",
            },
            "hardware": hw_meta,
            "history": history,
        }, f, indent=2)
    log_print(f"Saved metrics summary: {summary_path}")
    log_print("=" * 70)

    log_fp.close()


if __name__ == "__main__":
    main()
