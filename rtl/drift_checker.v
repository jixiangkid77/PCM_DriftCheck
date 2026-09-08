// Drift-aware group checker datapath (E6 spec section 3, frozen).
// Exact fixed point: r_int = (y_ck << LOGM) - CSC * sum(y_j); the only
// approximation in the checker lives in the f_band value (Prop 3).
module drift_checker #(
    parameter BY   = 8,   // signed column code width
    parameter LOGM = 4,   // log2 group size
    parameter CSC  = 1,   // checksum scale c (1 or 3)
    parameter FB   = 3    // fractional threshold bits (f in 1/2^FB code units)
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        en,
    input  wire signed [BY-1:0]        y_ck,
    input  wire [(1<<LOGM)*BY-1:0]     y_flat,
    input  wire [BY+LOGM+1+FB:0]       f_band,  // unsigned, (code units * m) << FB
    output reg                         alarm,
    output wire [BY+LOGM+1:0]          r_abs
);
    localparam M  = (1 << LOGM);
    localparam WS = BY + LOGM;      // sum width
    localparam WR = BY + LOGM + 2;  // headroom for CSC = 3

    integer i;
    reg signed [WS-1:0] s_acc;
    always @* begin
        s_acc = {WS{1'b0}};
        for (i = 0; i < M; i = i + 1)
            s_acc = s_acc + {{LOGM{y_flat[i*BY+BY-1]}}, y_flat[i*BY +: BY]};
    end

    wire signed [WR-1:0] s_ext = {{(WR-WS){s_acc[WS-1]}}, s_acc};
    wire signed [WR-1:0] lhs   = {{(WR-BY-LOGM){y_ck[BY-1]}}, y_ck, {LOGM{1'b0}}};
    wire signed [WR-1:0] rhs   = (CSC == 3) ? ((s_ext <<< 1) + s_ext) : s_ext;
    wire signed [WR-1:0] r     = lhs - rhs;
    wire [WR-1:0] rabs_w       = r[WR-1] ? (~r + 1'b1) : r;
    assign r_abs = rabs_w;

    wire [WR+FB-1:0] rabs_x = {rabs_w, {FB{1'b0}}};
    always @(posedge clk or negedge rst_n)
        if (!rst_n)      alarm <= 1'b0;
        else if (en)     alarm <= (rabs_x > f_band);
endmodule
