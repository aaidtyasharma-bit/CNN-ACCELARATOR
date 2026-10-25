# Standard Constraints File (.sdc) for CNN Accelerator
# Clock target: 200 MHz (Period: 5.0 ns)

current_design cnn_accelerator_top

# 1. Primary Clock Definition (50 MHz / Period: 20.000 ns)
create_clock -name clk -period 20.000 [get_ports clk]
set_clock_uncertainty 0.200 [get_clocks clk]
set_clock_transition 0.100 [get_clocks clk]

# 2. Input Delays (2.0 ns)
set_input_delay -clock clk -max 2.000 [get_ports {rst_n start mode_fc in_valid last_element pool_enable}]
set_input_delay -clock clk -min 0.200 [get_ports {rst_n start mode_fc in_valid last_element pool_enable}]

set_input_delay -clock clk -max 2.000 [get_ports {bias_in[*] weight_in[*] data_in[*]}]
set_input_delay -clock clk -min 0.200 [get_ports {bias_in[*] weight_in[*] data_in[*]}]

set_input_delay -clock clk -max 2.000 [get_ports {pool_p00[*] pool_p01[*] pool_p10[*] pool_p11[*]}]
set_input_delay -clock clk -min 0.200 [get_ports {pool_p00[*] pool_p01[*] pool_p10[*] pool_p11[*]}]

# 3. Output Delays (2.0 ns)
set_output_delay -clock clk -max 2.000 [get_ports {busy done}]
set_output_delay -clock clk -min 0.200 [get_ports {busy done}]

set_output_delay -clock clk -max 2.000 [get_ports {acc_out[*] act_out_relu[*] pool_out[*]}]
set_output_delay -clock clk -min 0.200 [get_ports {acc_out[*] act_out_relu[*] pool_out[*]}]

# 4. Driving Cell & Load Capacitance
set_driving_cell -lib_cell sky130_fd_sc_hd__buf_2 [all_inputs]
set_load 0.05 [all_outputs]
