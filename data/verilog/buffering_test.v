module TOP ( );
  supply0 VSS;
  supply1 VDD;
  
BLK U1  (a, b);
BLK U2  (b, c);
BLK U3  (c, a);
BLK U4  (a, a);
BLK U5  (a, b);
BLK U6  (b, c);
BLK U7  (c, a);
BLK U8  (a, a);
BLK U9  (a, b);
BLK U10 (b, c);
BLK U11 (c, a);
BLK U12 (a, a);
BLK U13 (a, b);
BLK U14 (b, c);
BLK U15 (c, a);
BLK U16 (a, a);
BLK U17 (a, b);
BLK U18 (VSS, c);
BLK U19 (VDD, a);
BLK U20 (a, a);
 

endmodule


