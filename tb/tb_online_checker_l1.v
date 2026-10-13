`timescale 1ns/1ps
// Bit-true vector testbench for online_checker_l1 (E8.2 amendment A-1).
// Vector word: {exp_alarm, exp_rabs[WR-1:0], y_ck[BY-1:0], y_flat}
module tb_online_checker_l1;
    parameter BY = 8; parameter LOGM = 4;
    parameter S = `SHIFT; parameter KF = 6; parameter K1 = `K2;
    localparam M = (1<<LOGM); localparam WR = BY+LOGM+2;
    localparam TOTW = 1 + WR + BY + M*BY;
    reg [TOTW-1:0] tv [0:`NVEC-1];
    reg clk = 0, rst_n = 0, en = 1, load = 0;
    reg signed [BY-1:0] y_ck; reg [M*BY-1:0] y_flat;
    reg [WR+S:0] v_init;
    wire alarm; wire [WR-1:0] r_abs;
    online_checker_l1 #(.BY(BY), .LOGM(LOGM), .S(S), .KF(KF), .K1(K1)) dut (
        .clk(clk), .rst_n(rst_n), .en(en), .load(load), .y_ck(y_ck), .y_flat(y_flat),
        .m_init(v_init), .alarm(alarm), .r_abs(r_abs));
    integer i, bad, n_al; reg exp_a; reg [WR-1:0] exp_r;
    initial begin
        $readmemh(`VEC, tv);
        bad = 0; n_al = 0; v_init = `VINIT;
        load = 1; en = 0; #1 clk = 1; #1 clk = 0; load = 0; en = 1;
        #1 rst_n = 1;
        for (i = 0; i < `NVEC; i = i + 1) begin
            {exp_a, exp_r, y_ck, y_flat} = tv[i];
            #1;
            if (r_abs !== exp_r) begin bad = bad + 1;
                if (bad < 5) $display("MISMATCH rabs i=%0d got=%0d exp=%0d", i, r_abs, exp_r); end
            clk = 1; #1; clk = 0;
            n_al = n_al + alarm;
            if (alarm !== exp_a) begin bad = bad + 1;
                if (bad < 5) $display("MISMATCH alarm i=%0d got=%0d exp=%0d", i, alarm, exp_a); end
        end
        $display("RESULT %0d mismatches of %0d vectors, %0d alarms", bad, `NVEC, n_al);
        $finish;
    end
endmodule
