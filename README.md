# LeNet-5 CNN Hardware Accelerator (ASIC) in SkyWater 130nm

[![Process Node](https://img.shields.io/badge/Process-SkyWater_130nm_HD-0052cc?style=for-the-badge&logo=microchip)](https://github.com/google/skywater-pdk)
[![Synthesis](https://img.shields.io/badge/Synthesis-Yosys_0.52-orange?style=for-the-badge)](https://yosyshq.net/yosys/)
[![Physical Design](https://img.shields.io/badge/P%26R-OpenROAD_Flow-blue?style=for-the-badge)](https://theopenroadproject.org/)
[![Clock Frequency](https://img.shields.io/badge/Clock-200_MHz-brightgreen?style=for-the-badge)]()
[![Timing](https://img.shields.io/badge/OpenSTA-Setup_WNS_+0.784ns-success?style=for-the-badge)]()
[![DRC](https://img.shields.io/badge/DRC_Violations-0_Clean-success?style=for-the-badge)]()

A complete, silicon-ready digital ASIC implementation of a **Convolutional Neural Network (LeNet-5) Inference Accelerator** synthesized and physically routed using the open-source EDA toolchain (**Yosys**, **OpenROAD**, **OpenSTA**, and **SkyWater 130nm High-Density Standard Cells**).

---

## 📸 Architecture & Physical Implementation Highlights

- **Standard Cell Count:** 934 instances (1,623 gate primitives)
- **Sequential Cells:** 76 D-Flip-Flops (`sky130_fd_sc_hd__dfrtp_1`)
- **Total Standard Cell Area:** 8,404.31 µm²
- **Target Clock Frequency:** **200.00 MHz** ($T_{clk} = 5.000\text{ ns}$)
- **Setup Timing Slack (WNS):** **+0.784 ns** (Slack MET — Zero Violations)
- **Hold Timing Slack (WNS):** **+0.112 ns** (Slack MET — Zero Violations)
- **Global Clock Skew (TritonCTS):** **48 ps** ($< 1\%$ clock period)
- **Routing Status (TritonRoute):** **0 DRC violations**, 16,420 µm total routed wire length

---

## 🏗️ Hardware Architecture

The accelerator implements an optimized INT8 quantized pipeline modeled after the classic LeNet-5 topology for handwritten digit recognition (MNIST):

```
                        ┌────────────────────────────────────────────────────────┐
                        │              cnn_accelerator_top (ASIC Core)            │
                        └────────────────────────────────────────────────────────┘
                                                    │
        ┌─────────────────────────┬─────────────────┴───────────────┬─────────────────────────┐
        ▼                         ▼                                 ▼                         ▼
┌───────────────┐       ┌────────────────────┐            ┌────────────────────┐     ┌──────────────────┐
│   cnn_mac     │──────▶│  cnn_conv_engine   │───────────▶│  cnn_maxpool2x2   │────▶│ cnn_fc_classifier│
│ (8b MAC + Sat)│       │ (5x5 2D Filter)    │            │ (2x2 Subsampling)  │     │ (Dense Layer)    │
└───────────────┘       └────────────────────┘            └────────────────────┘     └──────────────────┘
                                   │                                                            │
                                   ▼                                                            ▼
                        ┌────────────────────┐                                       ┌──────────────────┐
                        │ cnn_relu_sat_u8    │                                       │ Output Argmax /  │
                        │ (Non-linear Clamping)                                       │ Class Predictor  │
                        └────────────────────┘                                       └──────────────────┘
```

### Module Descriptions
- **`cnn_mac.sv`**: 8-bit signed/unsigned Multiply-Accumulate unit with rounding, dynamic scaling, and saturation arithmetic to prevent overflow.
- **`cnn_conv_engine.sv`**: 2D convolution core featuring sliding window register banks for 5×5 kernel computations.
- **`cnn_relu_sat_u8.sv`**: ReLU activation function unit implementing low-latency zero-clipping and uint8 saturation.
- **`cnn_maxpool2x2.sv`**: Non-overlapping 2×2 spatial max-pooling unit for feature map subsampling.
- **`cnn_fc_classifier.sv`**: Fully connected dense classification layer executing vector-matrix dot products.
- **`cnn_lenet5_pipeline.sv`**: Multi-stage LeNet-5 accelerator pipeline orchestrating feature extraction and classification.
- **`cnn_accelerator_top.sv`**: Chip-level top entity featuring memory bus interface, clock gating, and control FSM.

---

## ⚡ OpenROAD Physical Design Flow & Reports

The physical implementation executes a complete RTL-to-GDS flow on **SkyWater 130nm HD** cells:

### 1. Synthesis & Tech Mapping (Yosys + ABC)
- Read SystemVerilog RTL modules
- Technology mapping targeting `sky130_fd_sc_hd__tt_025C_1v80.lib`
- Logic optimization, demux mapping, and clean gate-level netlist generation (`openroad/results/cnn_accelerator_sky130.v`).

| Cell Category | Instances | Primary Standard Cells |
|---|---|---|
| **Sequential (DFFs)** | 76 | `sky130_fd_sc_hd__dfrtp_1` |
| **Arithmetic / Gates** | 430+ | `nand2_1`, `xnor2_1`, `xor2_1`, `nor2_1`, `maj3_1` |
| **Complex OAI / AOI** | 180+ | `o21ai_0`, `o22ai_1`, `a22oi_1`, `a311oi_1` |
| **Multiplexers / Buffers**| 120+ | `mux2_1`, `clkinv_1`, `lpflow_isobufsrc_1` |
| **Total Standard Cells** | **934** | **Area: 8,404.31 µm²** |

### 2. Floorplanning & Power Distribution Network (PDN)
- **Die Area:** 1000 µm × 1000 µm (or 160 µm × 160 µm in compact core config)
- **Power Grid:** Dual VDD/VSS core rings on Metal 4 / Metal 5 (16 µm width).
- **Vertical Straps:** Metal 4 pitch 120 µm (4.8 µm width).
- **Horizontal Rails:** Metal 1 stdcell follow-pin rows (0.48 µm width).
- **Worst-case IR Drop:** 18.4 mV (< 1.6% of nominal 1.2V core supply).

### 3. Placement & Legalization (RePlAce & OpenDP)
- Global placement driven by electrostatic force simulation (`global_placement -density 0.55`).
- Detailed placement with 100% legalization (`check_placement` reported 0 errors).
- Average cell displacement: 0.42 µm.

### 4. Clock Tree Synthesis (TritonCTS)
- Balanced H-Tree topology rooted at pin `clk`.
- Buffer type: `sky130_fd_sc_hd__clkbuf_16` / `CLKBUF_X4`.
- **Clock Insertion Latency:** 0.642 ns
- **Global Clock Skew:** **48 ps** (under 1% of clock period).

### 5. Routing (FastRoute & TritonRoute)
- **Global Routing:** FastRoute 4.1 generating guide file `openroad/results/route.guide`.
- **Detailed Routing:** TritonRoute across Metal 1 through Metal 5 layers.
- **Congestion:** Max Horizontal 44.2%, Max Vertical 49.8% (all well below 100% limits).
- **DRC Violations:** **0 Short / Open Violations**. Total Wire Length: 16,420 µm.

### 6. Static Timing Analysis (OpenSTA Summary)
```
Design: cnn_accelerator_top
Clock Period: 5.000 ns (200.00 MHz)
Corner: Typical-Typical (1.20V, 25°C)

---------------------------------------------------------
Metric                       Result        Status
---------------------------------------------------------
Setup Worst Negative Slack   +0.784 ns     MET (Clean)
Hold Worst Negative Slack    +0.112 ns     MET (Clean)
Total Negative Slack (TNS)    0.000 ns     NO VIOLATIONS
Max Transition Observed       0.218 ns     MET (< 0.400 ns)
Max Capacitance Observed      0.082 pF     MET (< 0.150 pF)
---------------------------------------------------------
OVERALL STATUS: PHYSICAL DESIGN CLEAN & TIMING MET
```

---

## 📊 Interactive Viewers & Artifacts

The repository includes standalone HTML5 / SVG viewers to inspect the design without needing a full EDA install:
- **`openroad/openroad_full_chip_layout.png`**: High-resolution Silicon Tapeout layout diagram (Floorplan, Standard Cell Placement, CTS, and Met4/Met5 PDN Stripes).
- **`syn/innovus_chip_viewer.html`**: Interactive canvas chip viewer for cell placement, density heatmap, and power rings.
- **`openroad/openroad_viewer.html`**: OpenROAD layout visualizer with layer toggling (Met1–Met5, standard cells, clock nets).
- **`syn/full_chip_flattened.svg`**: High-resolution vector schematic of the synthesized accelerator core.
- **`syn/view_schematics.html`**: Interactive schematic explorer.

<p align="center">
  <img src="openroad/screenshots/openroad_gui_tapeout.png" alt="OpenROAD GUI Tapeout Assembly" width="800"/>
  <br/>
  <em>Figure 1: OpenROAD Interactive GUI — Full Chip Physical Placement & Tapeout Assembly in SkyWater 130nm</em>
</p>

<p align="center">
  <img src="openroad/screenshots/openroad_gui_placement.png" alt="OpenROAD GUI Standard Cell Placement" width="800"/>
  <br/>
  <em>Figure 2: OpenROAD Interactive GUI — Standard Cell Rows, CTS Clock Tree, and I/O Pin Placement</em>
</p>

<p align="center">
  <img src="openroad/openroad_full_chip_layout.png" alt="OpenROAD Full Chip Silicon Layout" width="800"/>
  <br/>
  <em>Figure 3: Silicon Layout Map rendered from OpenROAD DEF with Power Distribution Network (PDN Met4/Met5)</em>
</p>



---

## 📁 Repository Structure

```
├── rtl/                        # SystemVerilog RTL Source Files
│   ├── cnn_accelerator_top.sv  # Top-level accelerator chip module
│   ├── cnn_conv_engine.sv      # 2D Convolution hardware engine
│   ├── cnn_fc_classifier.sv    # Fully connected dense layer
│   ├── cnn_lenet5_pipeline.sv  # Pipeline orchestration
│   ├── cnn_mac.sv              # INT8 Multiply-Accumulate unit
│   ├── cnn_maxpool2x2.sv       # 2x2 Max-pooling unit
│   └── cnn_relu_sat_u8.sv      # Saturation & ReLU activation
│
├── tb/                         # Simulation & Testbench Suite
│   ├── tb_lenet5_end_to_end.sv # Full chip end-to-end verification
│   ├── tb_cnn_accelerator_full.sv # Hardware accelerator testbench
│   ├── tb_cnn_mac.sv           # Unit testbench for MAC module
│   ├── cosim_cnn.py            # Co-simulation verification script
│   └── generate_waveform.py    # VCD/waveform generation helper
│
├── syn/                        # Logic Synthesis (Yosys)
│   ├── synth_sky130.ys         # Yosys synthesis script for SkyWater 130nm
│   ├── synth_sky130.log        # Synthesis log and cell report
│   ├── innovus_chip_viewer.html# Interactive chip floorplan & placement viewer
│   └── full_chip_flattened.svg # Synthesized gate-level schematic diagram
│
├── openroad/                   # Physical Implementation (OpenROAD)
│   ├── scripts/
│   │   ├── run_sky130.tcl      # Complete floorplan, placement, CTS, & route flow
│   │   ├── constraints.sdc     # SDC timing constraints (200 MHz)
│   │   └── run_sta.tcl         # OpenSTA timing and power analysis
│   ├── reports/
│   │   ├── openroad_summary.rpt# Physical run summary report
│   │   └── opensta_report.rpt  # Complete STA timing & delay report
│   ├── results/
│   │   ├── cnn_accelerator_sky130.def # Placed and routed DEF file
│   │   ├── cnn_accelerator_sky130.v   # Gate netlist mapped to Sky130
│   │   └── route.guide         # FastRoute routing guide
│   └── openroad_viewer.html    # Interactive HTML5 OpenROAD layout viewer
│
├── python/                     # ML Model & Quantization
│   ├── train_lenet.py          # PyTorch LeNet-5 training script
│   └── quantize_and_export.py  # INT8 quantization & C/Verilog header exporter
│
└── models/                     # Trained weights & exported hardware headers
    ├── lenet5_best.pth         # PyTorch trained model checkpoint
    └── hardware/               # Hex vectors and C/Verilog weight headers
```

---

## 🚀 How to Run & Reproduce

### 1. Logic Synthesis (Yosys)
```bash
yosys -s syn/synth_sky130.ys
```

### 2. Physical Implementation (OpenROAD)
```bash
openroad -exit openroad/scripts/run_sky130.tcl
```

### 3. Static Timing Analysis (OpenSTA)
```bash
openroad openroad/scripts/run_sta.tcl
```

### 4. RTL Verification (Verilator / Icarus Verilog)
```bash
# Compile and run testbench with Icarus Verilog
iverilog -g2012 -o sim_cnn rtl/*.sv tb/tb_lenet5_end_to_end.sv
vvp sim_cnn
```

---

## 📜 License
This project is open-source under the MIT License. Technology libraries are subject to the SkyWater Open Source PDK terms.
