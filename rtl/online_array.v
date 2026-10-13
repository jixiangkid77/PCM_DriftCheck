// Online variance-adaptive checker pooled over the K = 2^LOGK groups of one
// array: E8.2 cost reference for the published adaptive-threshold family.
// Each group forms r^2 as in online_checker; one estimator per array is fed
// with the mean of the K squares, v <- v + (mean_k r_k^2 - v) / 2^S, and every
// group compares its r^2 against the shared product k^2 * v.
module online_array #(
    parameter BY   = 8,
    parameter LOGM = 4,
    parameter LOGK = 3,
    parameter S    = 6,
    parameter KF   = 6,
    parameter K2   = 383
)(
    input  wire                                   clk,
    input  wire                                   rst_n,
    input  wire                                   en,
    input  wire                                   load,
    input  wire [(1<<LOGK)*BY-1:0]                y_ck_flat,
    input  wire [(1<<LOGK)*(1<<LOGM)*BY-1:0]      y_flat,
    input  wire [2*(BY+LOGM+2)+S:0]               v_init,
    output reg  [(1<<LOGK)-1:0]                   alarm
);
    localparam K  = (1 << LOGK);
    localparam M  = (1 << LOGM);
    localparam WS = BY + LOGM;
    localparam WR = BY + LOGM + 2;
    localparam WQ = 2 * WR;
    localparam WV = WQ + S + 1;

    wire [K*WQ-1:0] q_flat;
    genvar g;
    generate
        for (g = 0; g < K; g = g + 1) begin : grp
            integer i;
            reg signed [WS-1:0] s_acc;
            always @* begin
                s_acc = {WS{1'b0}};
                for (i = 0; i < M; i = i + 1)
                    s_acc = s_acc + {{LOGM{y_flat[(g*M+i)*BY+BY-1]}}, y_flat[(g*M+i)*BY +: BY]};
            end
            wire signed [WR-1:0] s_ext = {{(WR-WS){s_acc[WS-1]}}, s_acc};
            wire signed [BY-1:0] yck   = y_ck_flat[g*BY +: BY];
            wire signed [WR-1:0] lhs   = {{(WR-BY-LOGM){yck[BY-1]}}, yck, {LOGM{1'b0}}};
            wire signed [WR-1:0] r     = lhs - s_ext;
            wire [WR-1:0] rabs_w       = r[WR-1] ? (~r + 1'b1) : r;
            assign q_flat[g*WQ +: WQ]  = rabs_w * rabs_w;
        end
    endgenerate

    integer j;
    reg [WQ+LOGK-1:0] q_sum;
    always @* begin
        q_sum = {(WQ+LOGK){1'b0}};
        for (j = 0; j < K; j = j + 1)
            q_sum = q_sum + q_flat[j*WQ +: WQ];
    end
    wire [WQ-1:0] q_mean = q_sum >> LOGK;

    reg  [WV-1:0] v;
    wire [WV+15:0] rhs_c = v * K2;

    integer k;
    always @(posedge clk or negedge rst_n)
        if (!rst_n)  alarm <= {K{1'b0}};
        else if (en)
            for (k = 0; k < K; k = k + 1)
                alarm[k] <= ({q_flat[k*WQ +: WQ], {(S+KF){1'b0}}} > rhs_c);

    always @(posedge clk)
        if (load)    v <= v_init;
        else if (en) v <= v + q_mean - (v >> S);
endmodule
