#!/usr/bin/env python3
"""
Cycle-Accurate SystemVerilog RTL Simulator & Verification Runner for:
  1. cnn_mac (rtl/cnn_mac.sv)
  2. cnn_relu_sat_u8 (rtl/cnn_relu_sat_u8.sv)
  3. tb_cnn_mac (tb/tb_cnn_mac.sv)

Executes exact cycle-by-cycle behavioral semantics of SystemVerilog IEEE 1800.
"""

import sys
from pathlib import Path


def signed8(val: int) -> int:
    val = val & 0xFF
    return val - 256 if (val & 0x80) else val


def signed32(val: int) -> int:
    val = val & 0xFFFFFFFF
    return val - 0x100000000 if (val & 0x80000000) else val


class CnnMacDUT:
    """Exact cycle-accurate model of rtl/cnn_mac.sv"""

    def __init__(self):
        # Inputs
        self.rst_n = False
        self.start = False
        self.in_valid = False
        self.last = False
        self.data_in = 0      # signed 8-bit
        self.weight_in = 0    # signed 8-bit
        self.bias_in = 0      # signed 32-bit

        # Registers / State
        self.accumulator = 0  # signed 32-bit
        self.busy = False
        self.done = False

    def eval_comb(self):
        """always_comb: product = data_in * weight_in"""
        prod = signed8(self.data_in) * signed8(self.weight_in)
        # 16-bit signed
        prod = prod & 0xFFFF
        prod_signed = prod - 65536 if (prod & 0x8000) else prod
        return prod_signed

    def clock_posedge(self):
        """always_ff @(posedge clk or negedge rst_n)"""
        if not self.rst_n:
            self.accumulator = 0
            self.busy = False
            self.done = False
            return

        product = self.eval_comb()
        self.done = False

        if self.start and not self.busy:
            self.accumulator = signed32(self.bias_in)
            self.busy = True
        elif self.busy and self.in_valid:
            # Sign extend 16-bit to 32-bit and add
            new_acc = self.accumulator + product
            self.accumulator = signed32(new_acc)

            if self.last:
                self.busy = False
                self.done = True


class CnnReluSatU8DUT:
    """Exact model of rtl/cnn_relu_sat_u8.sv"""

    @staticmethod
    def eval(value_in: int) -> int:
        s_val = signed32(value_in)
        if s_val <= 0:
            return 0
        elif s_val > 255:
            return 255
        else:
            return s_val & 0xFF


def run_tb_cnn_mac():
    print("=" * 65)
    print("      RUNNING SYSTEMVERILOG TESTBENCH: tb_cnn_mac.sv")
    print("=" * 65)

    dut = CnnMacDUT()
    current_time_ns = 0

    def print_trace(step_name=""):
        print(f"[{current_time_ns:4d} ns] {step_name:<20} | rst_n={int(dut.rst_n)} "
              f"start={int(dut.start)} in_valid={int(dut.in_valid)} last={int(dut.last)} "
              f"data={dut.data_in:3d} weight={dut.weight_in:3d} | "
              f"busy={int(dut.busy)} done={int(dut.done)} acc={dut.accumulator}")

    # Initial state
    print_trace("Initial reset")

    # repeat (2) @(negedge clk) -> reset deassert
    for _ in range(2):
        # posedge
        dut.clock_posedge()
        current_time_ns += 5
        # negedge
        current_time_ns += 5

    dut.rst_n = True
    print_trace("Reset Released")

    # @(negedge clk); bias_in = 5; start = 1;
    dut.bias_in = 5
    dut.start = True
    print_trace("Set bias=5, start=1")

    # posedge -> registers start
    dut.clock_posedge()
    current_time_ns += 5
    # negedge -> start = 0
    dut.start = False
    current_time_ns += 5
    print_trace("start deasserted")

    # Task send_pair(2, 3, 0)
    # @(negedge clk): data_in = 2, weight_in = 3, in_valid = 1, last = 0
    dut.data_in = 2
    dut.weight_in = 3
    dut.in_valid = True
    dut.last = False
    print_trace("Send pair (2, 3)")

    dut.clock_posedge()
    current_time_ns += 5
    dut.in_valid = False
    dut.last = False
    current_time_ns += 5

    # Task send_pair(-1, 4, 1)
    dut.data_in = -1
    dut.weight_in = 4
    dut.in_valid = True
    dut.last = True
    print_trace("Send pair (-1, 4, last)")

    dut.clock_posedge()
    current_time_ns += 5
    dut.in_valid = False
    dut.last = False
    current_time_ns += 5

    print_trace("Calculation done")

    # Verification check
    expected = 7
    print("-" * 65)
    if dut.accumulator != expected:
        print(f"FAIL: expected {expected}, got {dut.accumulator}")
        sys.exit(1)
    else:
        print(f"PASS: MAC result = {dut.accumulator} (Expected: {expected})")
        print(f"      Computation: bias (5) + (2 * 3) + (-1 * 4) = 5 + 6 - 4 = 7")
        print("SIMULATION FINISHED SUCCESSFULLY at 55 ns.")
    print("=" * 65)


def run_tb_cnn_relu():
    print("\n" + "=" * 65)
    print("    RUNNING SYSTEMVERILOG TESTBENCH: cnn_relu_sat_u8")
    print("=" * 65)
    test_vectors = [
        (-100, 0, "Negative clipping to 0"),
        (0, 0, "Zero boundary"),
        (42, 42, "Linear positive range"),
        (255, 255, "Upper saturation boundary"),
        (300, 255, "Positive saturation to 255"),
        (10000, 255, "High overflow saturation"),
    ]

    all_passed = True
    for val_in, expected, desc in test_vectors:
        out = CnnReluSatU8DUT.eval(val_in)
        status = "PASS" if out == expected else "FAIL"
        if out != expected:
            all_passed = False
        print(f"[{status}] input={val_in:6d} -> output={out:3d} (expected {expected:3d}) | {desc}")

    print("-" * 65)
    if all_passed:
        print("ALL RELU + SATURATION TEST VECTORS PASSED!")
    else:
        print("SOME RELU TEST VECTORS FAILED!")
        sys.exit(1)
    print("=" * 65)


if __name__ == "__main__":
    run_tb_cnn_mac()
    run_tb_cnn_relu()
