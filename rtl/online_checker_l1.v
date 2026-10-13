// Online adaptive group checker tracking E|r| instead of E r^2: the cheapest
// form of the published adaptive-threshold family (no squarer). E8.2 cost
// reference only. m <- m + (|r| - m) / 2^S (S fractional bits); alarm when
// |r| > k1 * m with k1 = K1 / 2^KF fixed at design time; the alarm uses the
// state before the current inference updates it.
module online_checker_l1 #(
    parameter BY   = 8,
    parameter LOGM = 4,
    parameter S    = 6,
    parameter KF   = 6,
    parameter K1   = 185
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        en,
    input  wire                        load,
    input  wire signed [BY-1:0]        y_ck,
    input  wire [(1<<LOGM)*BY-1:0]     y_flat,
    input  wire [BY+LOGM+2+S:0]        m_init,
    output reg                         alarm,
    output wire [BY+LOGM+1:0]          r_abs
);
    localparam M  = (1 << LOGM);
    localparam WS = BY + LOGM;
    localparam WR = BY + LOGM + 2;
    localparam WM = WR + S + 1;

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

    reg  [WM-1:0] mv;
    wire [WR+S+KF-1:0] lhs_c = {rabs_w, {(S+KF){1'b0}}};
    wire [WM+15:0]     rhs_c = mv * K1;

    always @(posedge clk or negedge rst_n)
        if (!rst_n)       alarm <= 1'b0;
        else if (en)      alarm <= (lhs_c > rhs_c);

    always @(posedge clk)
        if (load)         mv <= m_init;
        else if (en)      mv <= mv + rabs_w - (mv >> S);
endmodule
