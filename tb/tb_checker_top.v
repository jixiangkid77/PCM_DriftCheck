`timescale 1ns/1ps
// Bit-true vector testbench for the self-contained top (Phase F).
// Vector word: {exp_alarm, exp_rabs, exp_f, seg[NSL-1:0], u[UB-1:0], y_ck, y_flat}
module tb_checker_top;
    parameter BY = 8; parameter LOGM = 4; parameter NSL = 5; parameter UB = 8; parameter FB = 3;
    localparam M = (1<<LOGM); localparam WR = BY+LOGM+2; localparam WF = WR+FB;
    localparam TOTW = 1 + WR + WF + NSL + UB + BY + M*BY;
    reg [TOTW-1:0] tv [0:`NVEC-1];
    reg clk = 0, rst_n = 0, en = 1;
    reg signed [BY-1:0] y_ck; reg [M*BY-1:0] y_flat;
    reg [NSL-1:0] seg; reg [UB-1:0] u;
    wire alarm; wire [WR-1:0] r_abs; wire [WF-1:0] f_dbg;
    checker_top #(.BY(BY), .LOGM(LOGM), .NSL(NSL), .UB(UB), .FB(FB)) dut (
        .clk(clk), .rst_n(rst_n), .en(en), .y_ck(y_ck), .y_flat(y_flat),
        .seg(seg), .u(u), .alarm(alarm), .r_abs(r_abs));
    assign f_dbg = dut.f_band;
    integer i, bad; reg exp_a; reg [WR-1:0] exp_r; reg [WF-1:0] exp_f;
    initial begin
        $readmemh("vectors_top.hex", tv);
        bad = 0; #2 rst_n = 1;
        for (i = 0; i < `NVEC; i = i + 1) begin
            {exp_a, exp_r, exp_f, seg, u, y_ck, y_flat} = tv[i];
            #1;
            if (r_abs !== exp_r) bad = bad + 1;
            if (f_dbg !== exp_f) begin bad = bad + 1;
                if (bad < 5) $display("MISMATCH f i=%0d got=%0d exp=%0d seg=%0d u=%0d", i, f_dbg, exp_f, seg, u); end
            clk = 1; #1; clk = 0;
            if (alarm !== exp_a) bad = bad + 1;
        end
        $display("RESULT %0d mismatches of %0d vectors", bad, `NVEC);
        $finish;
    end
endmodule
