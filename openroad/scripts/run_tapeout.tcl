# Complete Tapeout Physical Implementation Flow in OpenROAD
# Design: cnn_accelerator_top on SkyWater 130nm (sky130hd)

set TECH_LEF "pdk/sky130/sky130hd.tlef"
set SC_LEF   "pdk/sky130/sky130_fd_sc_hd_merged.lef"
set LIB_FILE "pdk/sky130/sky130_fd_sc_hd__tt_025C_1v80.lib"
set VERILOG  "openroad/results/cnn_accelerator_sky130.v"
set SDC_FILE "openroad/scripts/constraints.sdc"

puts "=========================================================================="
puts "            OpenROAD Full-Chip Tapeout Flow (SkyWater 130nm)             "
puts "=========================================================================="

# 1. Read Technology & Standard Cell Libraries
read_lef $TECH_LEF
read_lef $SC_LEF
read_liberty $LIB_FILE

# 2. Read Synthesized Gate Netlist & Link
read_verilog $VERILOG
link_design cnn_accelerator_top

# 3. Read SDC Timing Constraints
read_sdc $SDC_FILE

# 4. Floorplanning (Core Area: 160um x 160um)
initialize_floorplan -die_area "0 0 170 170" \
                     -core_area "15 15 155 155" \
                     -site unithd

source pdk/sky130/sky130hd.tracks

# 5. Place I/O Pins around the Core boundary
place_pins -hor_layers met3 -ver_layers met2

# 6. Synthesize Full-Chip Power Distribution Network (PDN VDD/VSS Mesh)
source pdk/sky130/sky130hd.pdn.tcl
pdngen

# 7. Global Placement (RePlAce)
global_placement -density 0.55 -overflow 0.1

# 8. Detailed Placement & Legalization (OpenDP)
detailed_placement
check_placement

# 9. Clock Tree Synthesis (TritonCTS)
clock_tree_synthesis -root_buf sky130_fd_sc_hd__clkbuf_16 -buf_list sky130_fd_sc_hd__clkbuf_16

# 10. Fill Empty Core Space with Standard Cell Decap / Fillers
filler_placement "sky130_fd_sc_hd__fill_1 sky130_fd_sc_hd__fill_2 sky130_fd_sc_hd__fill_4 sky130_fd_sc_hd__fill_8"

# 11. Global Routing (FastRoute)
global_route

# 12. Export Tapeout DEF
write_def openroad/results/cnn_accelerator_tapeout.def

puts "=========================================================================="
puts "       Tapeout Chip Assembly Complete! Launching OpenROAD GUI...         "
puts "=========================================================================="

gui::show
