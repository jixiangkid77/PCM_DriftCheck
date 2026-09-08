`timescale 1ns/1ps
// Bit-true vector testbench for drift_checker (Phase F).
// Vector word: {exp_alarm, exp_rabs[WR-1:0], f[WR-1:0], y_ck[BY-1:0], y_flat}
module tb_drift_checker;
    parameter BY = 8; parameter LOGM = 4;
`ifdef CSC3
    parameter CSC = 3; localparam VEC = "vectors_c3.hex";
`else
    parameter CSC = 1; localparam VEC = "vectors_c1.hex";
`endif
    parameter FB = 3;
    localparam M = (1<<LOGM); localparam WR = BY+LOGM+2; localparam WF = WR+FB;
    localparam TOTW = 1 + WR + WF + BY + M*BY;
    reg [TOTW-1:0] tv [0:`NVEC-1];
    reg clk = 0, rst_n = 0, en = 1;
    reg signed [BY-1:0] y_ck; reg [M*BY-1:0] y_flat; reg [WF-1:0] f_band;
    wire alarm; wire [WR-1:0] r_abs;
    drift_checker #(.BY(BY), .LOGM(LOGM), .CSC(CSC)) dut (
        .clk(clk), .rst_n(rst_n), .en(en), .y_ck(y_ck), .y_flat(y_flat),
        .f_band(f_band), .alarm(alarm), .r_abs(r_abs));
    defparam dut.FB = FB;
    integer i, bad; reg exp_a; reg [WR-1:0] exp_r;
    initial begin
        $readmemh(VEC, tv);
        bad = 0; #2 rst_n = 1;
        for (i = 0; i < `NVEC; i = i + 1) begin
            {exp_a, exp_r, f_band, y_ck, y_flat} = tv[i];
            #1;
            if (r_abs !== exp_r) begin bad = bad + 1;
                if (bad < 5) $display("MISMATCH rabs i=%0d got=%0d exp=%0d", i, r_abs, exp_r); end
            clk = 1; #1; clk = 0;
            if (alarm !== exp_a) begin bad = bad + 1;
                if (bad < 5) $display("MISMATCH alarm i=%0d got=%0d exp=%0d", i, alarm, exp_a); end
        end
        $display("RESULT %0d mismatches of %0d vectors", bad, `NVEC);
        $finish;
    end
endmodule
