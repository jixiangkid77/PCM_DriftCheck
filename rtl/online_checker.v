// Online variance-adaptive group checker: E8.2 cost reference for the
// published adaptive-threshold family, not the checker proposed here.
// The datapath up to |r| is the one of drift_checker (CSC = 1). The band input
// is replaced by an estimator held in the checker,
//     v <- v + (r^2 - v) / 2^S        (v kept with S fractional bits),
// and the alarm compares r^2 > k^2 * v with k^2 = K2 / 2^KF fixed at design
// time, the cheapest form of this alternative (no programmable multiplier).
// The alarm uses the state before the current inference updates it.
module online_checker #(
    parameter BY   = 8,
    parameter LOGM = 4,
    parameter S    = 6,
    parameter KF   = 6,
    parameter K2   = 339
)(
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              en,
    input  wire                              load,
    input  wire signed [BY-1:0]              y_ck,
    input  wire [(1<<LOGM)*BY-1:0]           y_flat,
    input  wire [2*(BY+LOGM+2)+S:0]          v_init,
    output reg                               alarm,
    output wire [BY+LOGM+1:0]                r_abs
);
    localparam M  = (1 << LOGM);
    localparam WS = BY + LOGM;
    localparam WR = BY + LOGM + 2;
    localparam WQ = 2 * WR;
    localparam WV = WQ + S + 1;

    integer i;
    reg signed [WS-1:0] s_acc;
    always @* begin
        s_acc = {WS{1'b0}};
        for (i = 0; i < M; i = i + 1)
            s_acc = s_acc + {{LOGM{y_flat[i*BY+BY-1]}}, y_flat[i*BY +: BY]};
    end
    wire signed [WR-1:0] s_ext = {{(WR-WS){s_acc[WS-1]}}, s_acc};
    wire signed [WR-1:0] lhs   = {{(WR-BY-LOGM){y_ck[BY-1]}}, y_ck, {LOGM{1'b0}}};
    wire signed [WR-1:0] r     = lhs - s_ext;
    wire [WR-1:0] rabs_w       = r[WR-1] ? (~r + 1'b1) : r;
    assign r_abs = rabs_w;

    wire [WQ-1:0] q = rabs_w * rabs_w;
    reg  [WV-1:0] v;
    wire [WQ+S+KF-1:0] lhs_c = {q, {(S+KF){1'b0}}};
    wire [WV+15:0]     rhs_c = v * K2;

    always @(posedge clk or negedge rst_n)
        if (!rst_n)       alarm <= 1'b0;
        else if (en)      alarm <= (lhs_c > rhs_c);

    always @(posedge clk)
        if (load)         v <= v_init;
        else if (en)      v <= v + q - (v >> S);
endmodule
