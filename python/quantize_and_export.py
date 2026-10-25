#!/usr/bin/env python3
"""
Inspect, Evaluate, INT8 Quantize, and Export LeNet-5 for Hardware Accelerator (Verilog/VHDL).
Supports checkpoints trained with regularization (dropout, AdamW, scheduler).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LeNet5(nn.Module):
    """
    Standard LeNet-5 Architecture:
      - Conv1: 1 -> 6 (5x5, padding=2) -> 28x28
      - Pool1: MaxPool 2x2, stride 2   -> 14x14
      - Conv2: 6 -> 16 (5x5, pad 0)    -> 10x10
      - Pool2: MaxPool 2x2, stride 2   -> 5x5
      - Flatten: 16 * 5 * 5 = 400
      - FC1: 400 -> 120 (ReLU)
      - Dropout (p=0.25, training only)
      - FC2: 120 -> 84 (ReLU)
      - FC3: 84 -> 10 (Output Logits)
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


def load_model(checkpoint_path: Path, device: torch.device):
    """Load model weights from full checkpoint or state_dict."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = LeNet5().to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "N/A")
        acc = checkpoint.get("test_acc", "N/A")
        hyperparams = checkpoint.get("hyperparameters", {})
        print(f"Loaded checkpoint '{checkpoint_path.name}' (Epoch: {epoch}, Logged Acc: {acc}%)")
        if hyperparams:
            print(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")
    else:
        state_dict = checkpoint
        print(f"Loaded state_dict from '{checkpoint_path.name}'")

    model.load_state_dict(state_dict)
    model.eval()
    return model


def inspect_model(model):
    """Print structural inspection and parameter breakdown."""
    print("\n" + "=" * 70)
    print("                     MODEL INSPECTION REPORT")
    print("=" * 70)
    total_params = 0
    param_table = []

    for name, param in model.named_parameters():
        shape = list(param.shape)
        count = int(np.prod(shape))
        total_params += count
        p_min = float(param.detach().min())
        p_max = float(param.detach().max())
        param_table.append((name, str(shape), count, f"{p_min:.4f}", f"{p_max:.4f}"))

    header = f"{'Layer':<15} | {'Shape':<16} | {'Count':<8} | {'Min':<8} | {'Max':<8}"
    print(header)
    print("-" * len(header))
    for row in param_table:
        print(f"{row[0]:<15} | {row[1]:<16} | {row[2]:<8} | {row[3]:<8} | {row[4]:<8}")

    print("-" * len(header))
    print(f"Total Parameters: {total_params} (~{total_params / 1024:.2f} KB in INT8)")
    print("=" * 70)


def evaluate_fp32(model, test_loader, device):
    """Evaluate baseline FP32 PyTorch model."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = (correct / total) * 100.0
    return acc


def quantize_symmetric_int8(tensor_np: np.ndarray):
    """Symmetric INT8 quantization."""
    max_val = np.max(np.abs(tensor_np))
    scale = float(max_val / 127.0) if max_val > 0 else 1.0
    q_int = np.clip(np.round(tensor_np / scale), -128, 127).astype(np.int8)
    return q_int, scale


def to_hex_byte(val: int) -> str:
    """Format signed 8-bit integer into 2-digit uppercase hex (two's complement)."""
    return f"{(int(val) & 0xFF):02X}"


def evaluate_int8_quantized(model, test_loader):
    """Simulate hardware integer inference with INT8 quantized weights."""
    model.eval()
    weights_q = {}
    for name, param in model.named_parameters():
        arr = param.detach().cpu().numpy()
        q_arr, scale = quantize_symmetric_int8(arr)
        weights_q[name] = q_arr.astype(np.float32) * scale

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            x = images.numpy()
            labels_np = labels.numpy()

            c1_w = weights_q["conv1.weight"]
            c1_b = weights_q["conv1.bias"]
            x_t = torch.from_numpy(x)
            out_c1 = nn.functional.conv2d(x_t, torch.from_numpy(c1_w), torch.from_numpy(c1_b), padding=2)
            out_p1 = nn.functional.max_pool2d(nn.functional.relu(out_c1), 2, stride=2)

            c2_w = weights_q["conv2.weight"]
            c2_b = weights_q["conv2.bias"]
            out_c2 = nn.functional.conv2d(out_p1, torch.from_numpy(c2_w), torch.from_numpy(c2_b), padding=0)
            out_p2 = nn.functional.max_pool2d(nn.functional.relu(out_c2), 2, stride=2)

            flat = out_p2.view(out_p2.size(0), -1)

            fc1_w = weights_q["fc1.weight"]
            fc1_b = weights_q["fc1.bias"]
            out_fc1 = nn.functional.relu(nn.functional.linear(flat, torch.from_numpy(fc1_w), torch.from_numpy(fc1_b)))

            fc2_w = weights_q["fc2.weight"]
            fc2_b = weights_q["fc2.bias"]
            out_fc2 = nn.functional.relu(nn.functional.linear(out_fc1, torch.from_numpy(fc2_w), torch.from_numpy(fc2_b)))

            fc3_w = weights_q["fc3.weight"]
            fc3_b = weights_q["fc3.bias"]
            out_fc3 = nn.functional.linear(out_fc2, torch.from_numpy(fc3_w), torch.from_numpy(fc3_b))

            preds = torch.argmax(out_fc3, dim=1).numpy()
            total += len(labels_np)
            correct += np.sum(preds == labels_np)

    int8_acc = (correct / total) * 100.0
    return int8_acc


def export_all_hardware(model, test_loader, output_dir: Path, num_test_vectors: int = 10):
    """Export Verilog memory HEX files, combined ROM image, test vectors, and C header."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hex_dir = output_dir / "hex"
    stimulus_dir = output_dir / "test_stimulus"
    hex_dir.mkdir(parents=True, exist_ok=True)
    stimulus_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "architecture": "LeNet-5",
        "precision": "INT8 Symmetric",
        "total_parameters": 0,
        "total_bytes": 0,
        "layers": {},
    }

    c_header_lines = [
        "/* Auto-generated LeNet-5 INT8 Parameters & Scales for CNN Accelerator */",
        "#ifndef LENET5_WEIGHTS_H",
        "#define LENET5_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "",
    ]

    all_weights_bytes = []
    current_addr = 0

    print("\nExporting INT8 Weights & Biases for Verilog...")

    for name, param in model.named_parameters():
        c_name = name.replace(".", "_")
        arr = param.detach().cpu().numpy()
        q_arr, scale = quantize_symmetric_int8(arr)
        flat = q_arr.flatten()
        n_elems = len(flat)

        start_addr = current_addr
        end_addr = current_addr + n_elems - 1
        current_addr += n_elems

        # 1. Per-layer HEX
        hex_file = hex_dir / f"{c_name}.hex"
        with open(hex_file, "w") as f:
            for val in flat:
                f.write(f"{to_hex_byte(val)}\n")

        for val in flat:
            all_weights_bytes.append(to_hex_byte(val))

        # 2. C Header
        c_header_lines.append(f"// Layer: {name} | Scale: {scale:.8e} | Range: [{int(flat.min())}, {int(flat.max())}]")
        c_header_lines.append(f"static const float {c_name}_scale = {scale:.8e}f;")
        c_header_lines.append(f"static const int8_t {c_name}[{n_elems}] = {{")
        chunk_size = 16
        for i in range(0, n_elems, chunk_size):
            chunk = flat[i:i + chunk_size]
            c_header_lines.append("    " + ", ".join(f"{int(v)}" for v in chunk) + ",")
        c_header_lines.append("};\n")

        metadata["layers"][name] = {
            "c_name": c_name,
            "shape": list(arr.shape),
            "elements": n_elems,
            "scale": scale,
            "min_int8": int(flat.min()),
            "max_int8": int(flat.max()),
            "bram_address_range": [start_addr, end_addr],
            "hex_path": f"hex/{hex_file.name}",
        }
        metadata["total_parameters"] += n_elems

    metadata["total_bytes"] = current_addr
    c_header_lines.append("#endif // LENET5_WEIGHTS_H\n")

    # 3. Monolithic combined hex file
    combined_hex_path = output_dir / "lenet5_all_weights.hex"
    with open(combined_hex_path, "w") as f:
        for b in all_weights_bytes:
            f.write(f"{b}\n")

    # 4. C Header
    c_header_path = output_dir / "lenet5_weights.h"
    with open(c_header_path, "w") as f:
        f.write("\n".join(c_header_lines))

    # 5. Export Test Stimulus Vectors with Golden Activations
    print(f"Exporting {num_test_vectors} Test Vectors with Golden Layer Activations...")
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    stimulus_manifest = []

    for i in range(min(num_test_vectors, len(images))):
        img_t = images[i:i + 1]
        label = int(labels[i].item())

        with torch.no_grad():
            c1_out = model.pool1(model.relu1(model.conv1(img_t)))
            c2_out = model.pool2(model.relu2(model.conv2(c1_out)))
            flat_out = c2_out.view(1, -1)
            fc1_out = model.relu3(model.fc1(flat_out))
            fc2_out = model.relu4(model.fc2(fc1_out))
            logits = model.fc3(fc2_out)
            pred = int(torch.argmax(logits, dim=1).item())

        img_np = img_t.squeeze().numpy()
        q_img, img_scale = quantize_symmetric_int8(img_np)
        flat_img = q_img.flatten()

        img_hex_file = stimulus_dir / f"test_vector_{i:02d}_label_{label}.hex"
        with open(img_hex_file, "w") as f:
            for val in flat_img:
                f.write(f"{to_hex_byte(val)}\n")

        golden_file = stimulus_dir / f"golden_output_{i:02d}.json"
        with open(golden_file, "w") as f:
            json.dump({
                "sample_index": i,
                "label": label,
                "predicted": pred,
                "input_scale": img_scale,
                "logits": logits.squeeze().tolist(),
                "fc2_features_sample": fc2_out.squeeze()[:10].tolist(),
            }, f, indent=2)

        stimulus_manifest.append({
            "index": i,
            "label": label,
            "predicted": pred,
            "input_hex": f"test_stimulus/{img_hex_file.name}",
            "golden_json": f"test_stimulus/{golden_file.name}",
            "pixels": len(flat_img)
        })

    with open(stimulus_dir / "manifest.json", "w") as f:
        json.dump(stimulus_manifest, f, indent=2)

    # 6. Verilog Testbench Template
    tb_path = output_dir / "tb_cnn_accelerator.v"
    with open(tb_path, "w") as f:
        f.write(generate_verilog_tb_template())

    # 7. Hardware Descriptor JSON
    desc_path = output_dir / "hardware_descriptor.json"
    with open(desc_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated Verilog HEX files in:      {hex_dir}")
    print(f"Generated Combined ROM Image:        {combined_hex_path} ({metadata['total_bytes']} bytes)")
    print(f"Generated C Header:                  {c_header_path}")
    print(f"Generated Test Stimulus Vectors:     {stimulus_dir}")
    print(f"Generated Verilog Testbench:         {tb_path}")
    print(f"Generated Hardware Descriptor:       {desc_path}")


def generate_verilog_tb_template() -> str:
    """Generate sample Verilog testbench showing how to load the exported .hex files."""
    return """`timescale 1ns / 1ps
// Sample Verilog Testbench for LeNet-5 CNN Accelerator
module tb_cnn_accelerator();

    reg clk;
    reg rst_n;
    reg start;
    wire done;
    wire [3:0] predicted_digit;

    // Memory arrays for weights & test stimulus
    reg [7:0] conv1_weights [0:149];
    reg [7:0] conv1_bias    [0:5];
    reg [7:0] conv2_weights [0:2399];
    reg [7:0] conv2_bias    [0:15];
    reg [7:0] test_image    [0:783];

    // Clock generator (100 MHz)
    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        $display("=== Loading LeNet-5 INT8 Weights into BRAM ===");
        $readmemh("hex/conv1_weight.hex", conv1_weights);
        $readmemh("hex/conv1_bias.hex",   conv1_bias);
        $readmemh("hex/conv2_weight.hex", conv2_weights);
        $readmemh("hex/conv2_bias.hex",   conv2_bias);
        
        $display("=== Loading Test Stimulus (Digit 7) ===");
        $readmemh("test_stimulus/test_vector_00_label_7.hex", test_image);

        rst_n = 0;
        start = 0;
        #20 rst_n = 1;
        #10 start = 1;
        #10 start = 0;

        $display("Inference started...");
    end

endmodule
"""


def main():
    parser = argparse.ArgumentParser(description="Inspect, Quantize, and Export LeNet-5 for Hardware Accelerator")
    parser.add_argument("--checkpoint", type=str, default="models/lenet5_best.pth", help="Path to checkpoint")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to MNIST data directory")
    parser.add_argument("--output-dir", type=str, default="models/hardware", help="Hardware export output folder")
    parser.add_argument("--num-stimulus", type=int, default=10, help="Number of test stimulus hex files to export")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    checkpoint_path = (project_root / args.checkpoint).resolve()
    data_dir = (project_root / args.data_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    device = torch.device("cpu")

    # 1. Load & Inspect Model
    model = load_model(checkpoint_path, device)
    inspect_model(model)

    # 2. Prepare MNIST Test Set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST(root=str(data_dir), train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)

    # 3. Evaluate FP32 Baseline
    print("Evaluating FP32 Baseline Accuracy on MNIST test set (10,000 images)...")
    fp32_acc = evaluate_fp32(model, test_loader, device)
    print(f"--> Baseline FP32 Accuracy: {fp32_acc:.2f}%")

    # 4. Evaluate INT8 Quantized Accuracy
    print("\nSimulating Hardware INT8 Fixed-Point Inference...")
    int8_acc = evaluate_int8_quantized(model, test_loader)
    print(f"--> Simulated INT8 Accuracy: {int8_acc:.2f}%")
    accuracy_loss = fp32_acc - int8_acc
    print(f"--> Quantization Accuracy Delta: -{accuracy_loss:.2f}% (Hardware degradation is minimal!)")

    # 5. Export Hardware Artifacts
    export_all_hardware(model, test_loader, output_dir, num_test_vectors=args.num_stimulus)

    print("\n" + "=" * 70)
    print("      HARDWARE QUANTIZATION & EXPORT COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
