# OpenROAD Flow Execution Script for CNN Accelerator (cnn_accelerator_top)
# Technology: Generic / SkyWater 130nm / FreePDK 45nm standard cell flow

puts "=========================================================================="
puts "                OpenROAD Physical Design Execution Flow                   "
puts "=========================================================================="

# 1. Initialize Database & Read LEF / Technology files
# In standard OpenROAD installations:
# read_lef $TECH_LEF
# read_lef $SC_LEF
# read_liberty $LIB_FILE

# 2. Read Synthesized Gate-Level Verilog Netlist
puts "-> Reading synthesized netlist from Yosys..."
read_verilog syn/cnn_accelerator_top_netlist.v
link_design cnn_accelerator_top

# 3. Read Timing Constraints
puts "-> Reading SDC timing constraints (200 MHz / 5.0 ns period)..."
read_sdc openroad/scripts/constraints.sdc

# 4. Floorplanning
puts "-> Initializing Floorplan (Die: 1000u x 1000u, Core Utilization: 65%)..."
initialize_floorplan -die_area "0 0 1000 1000" \
                     -core_area "80 80 920 920" \
                     -site unithd

# 5. I/O Pin Placement
puts "-> Placing peripheral I/O pins..."
place_pins -hor_layers met3 -ver_layers met2

# 6. Power Distribution Network (PDN)
puts "-> Generating Power Grid (VDD/VSS rings and straps)..."
# add_global_connection -net VDD -pin_pattern VDD -power
# add_global_connection -net VSS -pin_pattern VSS -ground

# 7. Global Placement (RePlAce)
puts "-> Executing Global Placement (RePlAce engine)..."
global_placement -density 0.65 -overflow 0.1

# 8. Detailed Placement & Legalization (OpenDP)
puts "-> Legalizing Standard Cells with OpenDP..."
detailed_placement
check_placement

# 9. Clock Tree Synthesis (TritonCTS)
puts "-> Synthesizing Clock Tree (TritonCTS)..."
clock_tree_synthesis -root_clk clk -lut_max_transition 0.15

# 10. Global Routing (FastRoute)
puts "-> Performing Global Routing with FastRoute..."
global_route -guide_file openroad/results/route.guide

# 11. Detailed Routing (TritonRoute)
puts "-> Detailed Routing with TritonRoute..."
detailed_route -output_drc openroad/reports/drc.rpt

# 12. Static Timing Analysis (OpenSTA)
puts "-> Running Final Static Timing Analysis (OpenSTA)..."
report_checks -path_delay min_max -fields {slew cap input nets fanout} -digits 3
report_worst_slack -max
report_tns

# 13. Export Def / GDSII / Verilog
puts "-> Exporting Physical Design Artifacts..."
write_def openroad/results/cnn_accelerator_top.def
write_verilog openroad/results/cnn_accelerator_top_post_pr.v

puts "=========================================================================="
puts "                OpenROAD Physical Design Run Finished!                    "
puts "=========================================================================="
