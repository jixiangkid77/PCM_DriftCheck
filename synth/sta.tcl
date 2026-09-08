# OpenSTA timing script (KID side, authoritative; E6 spec Phase S).
# Usage: sta synth/sta.tcl  (edit NETLIST/TOP as needed)
set LIB synth/sky130_fd_sc_hd__tt_025C_1v80.lib
set NETLIST synth/out/checker_top_8_4.v
set TOP checker_top
read_liberty $LIB
read_verilog $NETLIST
link_design $TOP
create_clock -name clk -period 10 [get_ports clk]
set_input_delay 0 -clock clk [all_inputs]
set_output_delay 0 -clock clk [all_outputs]
report_checks -path_delay max -digits 3
report_worst_slack -max
# Fmax = 1 / (10 ns - worst slack); record both numbers in the run log.
