# Interactive OpenROAD / OpenSTA GUI Launcher with Timing Analysis Panels

# 1. Load Standard Cell LEF & Liberty Models
read_lef pdk/sky130/sky130hd.tlef
read_lef pdk/sky130/sky130_fd_sc_hd_merged.lef
read_liberty pdk/sky130/sky130_fd_sc_hd__tt_025C_1v80.lib

# 2. Read Physical DEF (Placed, Legalized & Routed)
read_def openroad/results/cnn_accelerator_sky130.def

# 3. Read SDC Timing Constraints
read_sdc openroad/scripts/constraints.sdc

# 4. Run OpenSTA Setup & Build Timing Graph
check_setup
report_worst_slack -max
report_worst_slack -min

# 5. Open Graphical User Interface with Timing Path Explorer
gui::show
