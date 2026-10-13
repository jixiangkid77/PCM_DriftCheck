`timescale 1ns/1ps
// Bit-true vector testbench for online_array (E8.2).
// Vector word: {exp_alarm[K-1:0], y_ck_flat, y_flat}
module tb_online_array;
    parameter BY = 8; parameter LOGM = 4; parameter LOGK = 3;
    parameter S = `SHIFT; parameter KF = 6; parameter K2 = `K2;
    localparam M = (1<<LOGM); localparam K = (1<<LOGK); localparam WR = BY+LOGM+2;
    localparam TOTW = K + K*BY + K*M*BY;
    reg [TOTW-1:0] tv [0:`NVEC-1];
    reg clk = 0, rst_n = 0, en = 1, load = 0;
    reg [K*BY-1:0] y_ck_flat; reg [K*M*BY-1:0] y_flat;
    reg [2*WR+S:0] v_init;
    wire [K-1:0] alarm;
    online_array #(.BY(BY), .LOGM(LOGM), .LOGK(LOGK), .S(S), .KF(KF), .K2(K2)) dut (
        .clk(clk), .rst_n(rst_n), .en(en), .load(load), .y_ck_flat(y_ck_flat),
        .y_flat(y_flat), .v_init(v_init), .alarm(alarm));
    integer i, bad, n_al; reg [K-1:0] exp_a;
    initial begin
        $readmemh(`VEC, tv);
        bad = 0; n_al = 0; v_init = `VINIT;
        load = 1; en = 0; #1 clk = 1; #1 clk = 0; load = 0; en = 1;
        #1 rst_n = 1;
        for (i = 0; i < `NVEC; i = i + 1) begin
            {exp_a, y_ck_flat, y_flat} = tv[i];
            #1 clk = 1; #1; clk = 0;
            n_al = n_al + (alarm != 0);
            if (alarm !== exp_a) begin bad = bad + 1;
                if (bad < 5) $display("MISMATCH alarm i=%0d got=%b exp=%b", i, alarm, exp_a); end
        end
        $display("RESULT %0d mismatches of %0d vectors, %0d alarm cycles", bad, `NVEC, n_al);
        $finish;
    end
endmodule
