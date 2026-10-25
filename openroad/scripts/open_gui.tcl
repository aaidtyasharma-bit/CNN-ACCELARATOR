# OpenROAD GUI Launch Script for CNN Accelerator on Sky130
read_lef pdk/sky130/sky130hd.tlef
read_lef pdk/sky130/sky130_fd_sc_hd_merged.lef
read_liberty pdk/sky130/sky130_fd_sc_hd__tt_025C_1v80.lib
read_def openroad/results/cnn_accelerator_sky130.def
read_sdc openroad/scripts/constraints.sdc

gui::show
