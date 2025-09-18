module TOP ( input [3:0] in, output [3:0] out);
  supply0 VSS;
  supply1 VDD;
  
  wire [3:0] mid;
 
  //AND U0 (mid[0], in[1], in[2], in[3], VDD, VSS);
  //NOT U1 (.o(mid[1]), .i(in[1]), .pwr(VDD), .gnd(VSS));
  //BUF U2 (.o(mid[2]), .i(in[2]), .pwr(VDD), .gnd(VSS));
  //DUT U3 (out, mid, VDD, VSS);

  ONE U0 (a, b);
  TWO U1 (b, c);
  TRREE U2 (c, a);

endmodule

module AND(out, in1, in2, in3, VDD, VSS);
  output out;
  input in1, in2, in3 ;
  input VDD, VSS;
endmodule

