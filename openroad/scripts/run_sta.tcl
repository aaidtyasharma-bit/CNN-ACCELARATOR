# Comprehensive Static Timing Analysis (OpenSTA) for CNN Accelerator
# Standard Cell Library: SkyWater 130nm High-Density (sky130_fd_sc_hd)

set LIB_FILE "pdk/sky130/sky130_fd_sc_hd__tt_025C_1v80.lib"
set TECH_LEF "pdk/sky130/sky130hd.tlef"
set SC_LEF   "pdk/sky130/sky130_fd_sc_hd_merged.lef"
set DEF_FILE "openroad/results/cnn_accelerator_sky130.def"
set SDC_FILE "openroad/scripts/constraints.sdc"

puts "=========================================================================="
puts "                     OpenSTA: STATIC TIMING ANALYSIS                      "
puts "                Design: cnn_accelerator_top (CNN Core)                   "
puts "=========================================================================="

# 1. Read Technology & Standard Cell Libraries
read_liberty $LIB_FILE
read_lef $TECH_LEF
read_lef $SC_LEF

# 2. Read Placed & Legalized Physical Design
read_def $DEF_FILE

# 3. Read SDC Constraints
read_sdc $SDC_FILE

# 4. Check Unconstrained Paths & Design Integrity
puts "\n--------------------------------------------------------------------------"
puts " STAGE 1: SDC Setup & Clock Network Verification"
puts "--------------------------------------------------------------------------"
check_setup -verbose

# 5. Worst Setup Timing Paths (Reg-to-Reg, In-to-Reg, Reg-to-Out)
puts "\n--------------------------------------------------------------------------"
puts " STAGE 2: Worst Setup Timing Paths (Max Delay - 200 MHz Target)"
puts "--------------------------------------------------------------------------"
report_checks -path_delay max -fields {slew cap input nets fanout} -digits 4 -endpoint_count 10

# 6. Worst Hold Timing Paths (Min Delay)
puts "\n--------------------------------------------------------------------------"
puts " STAGE 3: Worst Hold Timing Paths (Min Delay / Race Conditions)"
puts "--------------------------------------------------------------------------"
report_checks -path_delay min -fields {slew cap input nets fanout} -digits 4 -endpoint_count 5

# 7. Overall Slack & Violations Summary
puts "\n--------------------------------------------------------------------------"
puts " STAGE 4: Global Slack & Violations Summary"
puts "--------------------------------------------------------------------------"
report_worst_slack -max
report_worst_slack -min
report_tns

# 8. Slew, Capacitance & Fanout DRC Checks
puts "\n--------------------------------------------------------------------------"
puts " STAGE 5: Electrical DRC Checks (Max Transition / Capacitance / Fanout)"
puts "--------------------------------------------------------------------------"
report_check_types -max_slew -max_capacitance -max_fanout -violators

# 9. Clock Network Characterization (CTS Insertion Delay & Skew)
puts "\n--------------------------------------------------------------------------"
puts " STAGE 6: Clock Skew and Clock Insertion Delay"
puts "--------------------------------------------------------------------------"
report_clock_skew

# 10. Maximum Operating Frequency (Fmax Calculation)
puts "\n--------------------------------------------------------------------------"
puts " STAGE 7: Maximum Operating Frequency (Fmax)"
puts "--------------------------------------------------------------------------"
report_clock_min_period

puts "=========================================================================="
puts "                OpenSTA Timing Analysis Complete!                         "
puts "=========================================================================="
exit
