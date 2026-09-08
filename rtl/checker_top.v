// Self-contained variant: PWL band evaluation + checker (E6 spec section 1).
module checker_top #(
    parameter BY   = 8,
    parameter LOGM = 4,
    parameter NSL  = 4,
    parameter UB   = 8,
    parameter FB   = 3
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    en,
    input  wire signed [BY-1:0]    y_ck,
    input  wire [(1<<LOGM)*BY-1:0] y_flat,
    input  wire [NSL-1:0]          seg,
    input  wire [UB-1:0]           u,
    output wire                    alarm,
    output wire [BY+LOGM+1:0]      r_abs
);
    wire [BY+LOGM+1+FB:0] f_band;
    pwl_band #(.WR(BY+LOGM+2+FB), .NSL(NSL), .UB(UB)) u_pwl (
        .seg(seg), .u(u), .f_band(f_band));
    wire alarm_r;
    drift_checker #(.BY(BY), .LOGM(LOGM), .CSC(1), .FB(FB)) u_chk (
        .clk(clk), .rst_n(rst_n), .en(en), .y_ck(y_ck), .y_flat(y_flat),
        .f_band(f_band), .alarm(alarm_r), .r_abs(r_abs));
    assign alarm = alarm_r;
endmodule
