// E6 scale-reference module (spec section 1, decision 4): m-column output
// latch bank, synthesized only as an area yardstick; not part of the checker.
module latch_bank #(parameter BY=8, LOGM=4)(input wire clk,
  input wire [(1<<LOGM)*BY-1:0] d, output reg [(1<<LOGM)*BY-1:0] qo);
  always @(posedge clk) qo <= d;
endmodule
