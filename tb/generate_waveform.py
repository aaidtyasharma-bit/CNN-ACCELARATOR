#!/usr/bin/env python3
"""
Waveform Generator & VCD Dumper for CNN Accelerator Simulation
Generates:
  1. Standard VCD (Value Change Dump) file: outputs/waveforms/cnn_mac_simulation.vcd
     (viewable in GTKWave or VS Code Waveform Viewer extensions).
  2. High-resolution graphical waveform diagram: outputs/waveforms/waveform_timing_diagram.png
  3. ASCII text-based timing timing chart in terminal.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def signed8(val: int) -> int:
    val = val & 0xFF
    return val - 256 if (val & 0x80) else val


def signed32(val: int) -> int:
    val = val & 0xFFFFFFFF
    return val - 0x100000000 if (val & 0x80000000) else val


def dump_vcd_and_plot():
    project_root = Path(__file__).resolve().parent.parent
    wave_dir = project_root / "outputs" / "waveforms"
    wave_dir.mkdir(parents=True, exist_ok=True)

    vcd_path = wave_dir / "cnn_mac_simulation.vcd"
    png_path = wave_dir / "waveform_timing_diagram.png"

    # Simulation signal history
    times = []
    signals = {
        "clk": [],
        "rst_n": [],
        "start": [],
        "in_valid": [],
        "last": [],
        "data_in": [],
        "weight_in": [],
        "accumulator": [],
        "busy": [],
        "done": [],
        "relu_out": [],
    }

    # Simulation variables
    clk = 0
    rst_n = 0
    start = 0
    in_valid = 0
    last = 0
    data_in = 0
    weight_in = 0
    bias_in = 5
    accumulator = 0
    busy = 0
    done = 0
    relu_out = 0

    pairs = [(2, 3), (-1, 4), (10, 2), (-3, 5), (8, -2)]
    pair_idx = 0

    t_ns = 0

    def record_step():
        times.append(t_ns)
        signals["clk"].append(clk)
        signals["rst_n"].append(rst_n)
        signals["start"].append(start)
        signals["in_valid"].append(in_valid)
        signals["last"].append(last)
        signals["data_in"].append(data_in)
        signals["weight_in"].append(weight_in)
        signals["accumulator"].append(accumulator)
        signals["busy"].append(busy)
        signals["done"].append(done)
        signals["relu_out"].append(relu_out)

    # 0 ns
    record_step()

    # Step through 12 clock cycles (10 ns period)
    for cycle in range(12):
        # 1. Rising edge @ +5ns
        t_ns += 5
        clk = 1

        # Clocked logic (always_ff)
        if not rst_n:
            accumulator = 0
            busy = 0
            done = 0
        else:
            done = 0
            if start and not busy:
                accumulator = bias_in
                busy = 1
            elif busy and in_valid:
                prod = signed8(data_in) * signed8(weight_in)
                accumulator = signed32(accumulator + prod)
                if last:
                    busy = 0
                    done = 1

        relu_out = 0 if accumulator <= 0 else (255 if accumulator > 255 else (accumulator & 0xFF))
        record_step()

        # 2. Falling edge @ +5ns
        t_ns += 5
        clk = 0

        # Testbench actions on negedge
        if cycle == 1:
            rst_n = 1
        elif cycle == 2:
            start = 1
            bias_in = 5
        elif cycle == 3:
            start = 0
            data_in, weight_in = pairs[pair_idx]
            in_valid = 1
            last = 0
            pair_idx += 1
        elif cycle == 4:
            data_in, weight_in = pairs[pair_idx]
            in_valid = 1
            last = 0
            pair_idx += 1
        elif cycle == 5:
            data_in, weight_in = pairs[pair_idx]
            in_valid = 1
            last = 0
            pair_idx += 1
        elif cycle == 6:
            data_in, weight_in = pairs[pair_idx]
            in_valid = 1
            last = 1
            pair_idx += 1
        elif cycle == 7:
            in_valid = 0
            last = 0
            data_in = 0
            weight_in = 0

        record_step()

    # --- 1. Write standard VCD file ---
    with open(vcd_path, "w") as f:
        f.write("$date\n   Oct 16, 2026\n$end\n")
        f.write("$version\n   CNN Accelerator VCD Generator\n$end\n")
        f.write("$timescale\n   1ns\n$end\n")
        f.write("$scope module tb_cnn_mac $end\n")
        f.write("$var wire 1 ! clk $end\n")
        f.write("$var wire 1 \" rst_n $end\n")
        f.write("$var wire 1 # start $end\n")
        f.write("$var wire 1 $ in_valid $end\n")
        f.write("$var wire 1 % last $end\n")
        f.write("$var wire 8 & data_in [7:0] $end\n")
        f.write("$var wire 8 ' weight_in [7:0] $end\n")
        f.write("$var wire 32 ( accumulator [31:0] $end\n")
        f.write("$var wire 1 ) busy $end\n")
        f.write("$var wire 1 * done $end\n")
        f.write("$var wire 8 + relu_out [7:0] $end\n")
        f.write("$upscope $end\n$enddefinitions $end\n$dumpvars\n")

        for i, t in enumerate(times):
            f.write(f"#{t}\n")
            f.write(f"{signals['clk'][i]}!\n")
            f.write(f"{signals['rst_n'][i]}\"\n")
            f.write(f"{signals['start'][i]}#\n")
            f.write(f"{signals['in_valid'][i]}$\n")
            f.write(f"{signals['last'][i]}%\n")
            f.write(f"b{signals['data_in'][i] & 0xFF:08b} &\n")
            f.write(f"b{signals['weight_in'][i] & 0xFF:08b} '\n")
            f.write(f"b{signals['accumulator'][i] & 0xFFFFFFFF:032b} (\n")
            f.write(f"{signals['busy'][i]})\n")
            f.write(f"{signals['done'][i]}*\n")
            f.write(f"b{signals['relu_out'][i] & 0xFF:08b} +\n")

    # --- 2. Generate graphical timing diagram (Matplotlib) ---
    fig, axes = plt.subplots(8, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor('#0d1117')

    time_arr = np.array(times)

    def style_ax(ax, label, data, is_bus=False, color='#58a6ff'):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

        if not is_bus:
            ax.step(time_arr, data, where='post', color=color, linewidth=2)
            ax.set_ylim(-0.2, 1.2)
            ax.set_yticks([0, 1])
        else:
            # Bus wave plot
            ax.step(time_arr, data, where='post', color='#bc8cff', linewidth=2)
            for i in range(0, len(time_arr) - 1, 2):
                val = data[i]
                if val != 0 or label == "acc":
                    ax.text(time_arr[i] + 2, val, str(val), color='#e6edf3', fontsize=9, fontweight='bold')
            ax.grid(True, color='#21262d', linestyle='--', alpha=0.5)

        ax.set_ylabel(label, color='#f0f6fc', fontsize=11, fontweight='bold', rotation=0, labelpad=40, ha='right')

    style_ax(axes[0], "clk", signals["clk"], color='#79c0ff')
    style_ax(axes[1], "rst_n", signals["rst_n"], color='#56d364')
    style_ax(axes[2], "start", signals["start"], color='#e3b341')
    style_ax(axes[3], "in_valid", signals["in_valid"], color='#d2a8ff')
    style_ax(axes[4], "data_in", signals["data_in"], is_bus=True)
    style_ax(axes[5], "weight_in", signals["weight_in"], is_bus=True)
    style_ax(axes[6], "acc (32b)", signals["accumulator"], is_bus=True)
    style_ax(axes[7], "done", signals["done"], color='#f85149')

    axes[-1].set_xlabel("Time (ns)", color='#f0f6fc', fontsize=12, labelpad=10)
    axes[0].set_title("CNN Accelerator Hardware MAC Timing Waveform (cnn_mac.sv)",
                      color='#f0f6fc', fontsize=15, pad=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig(png_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    print("=" * 65)
    print("           CNN ACCELERATOR HARDWARE WAVEFORM REPORT")
    print("=" * 65)
    print(f"1. VCD Waveform File (GTKWave format):")
    print(f"   -> {vcd_path}")
    print(f"\n2. High-Resolution Waveform Timing Diagram (PNG):")
    print(f"   -> {png_path}")
    print("=" * 65)

    # --- 3. ASCII Timing Chart in Terminal ---
    print("\n--- ASCII DIGITAL WAVEFORM DIAGRAM ---")
    print("Time:      0ns  10ns  20ns  30ns  40ns  50ns  60ns  70ns  80ns  90ns 100ns 110ns")
    print("clk:       _/~\\_/~\\_/~\\_/~\\_/~\\_/~\\_/~\\_/~\\_/~\\_/~\\_/~\\_")
    print("rst_n:     ____/~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("start:     ___________/~~~~\\____________________________")
    print("busy:      _________________/~~~~~~~~~~~~~~~~~~~~~\\_____")
    print("in_valid:  _________________/~~~~~~~~~~~~~~~~~~~~~\\_____")
    print("data_in:   =================<  2  >< -1  >< 10  >< -3  >=========")
    print("weight_in: =================<  3  ><  4  ><  2  ><  5  >=========")
    print("acc (32b): 00000000000000005< 11  ><  7  >< 27  >< 12  >=========")
    print("done:      _______________________________________/~~~~\\_")
    print("-----------------------------------------------------------------")
    print("Summary:")
    print("  Cycle 1-2: Reset pulse (rst_n deasserted)")
    print("  Cycle 3:   Start pulse loaded bias = 5")
    print("  Cycle 4:   bias(5) + (2 * 3) = 11")
    print("  Cycle 5:   11 + (-1 * 4) = 7")
    print("  Cycle 6:   7 + (10 * 2) = 27")
    print("  Cycle 7:   27 + (-3 * 5) = 12 (last=1 asserted, done goes HIGH)")
    print("=================================================================\n")


if __name__ == "__main__":
    dump_vcd_and_plot()
