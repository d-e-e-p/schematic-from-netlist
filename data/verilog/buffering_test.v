module TOP ( );
  supply0 VSS;
  supply1 VDD;
  
BLK U01  (a, b);
BLK U02  (b, c);
BLK U03  (c, a);
BLK U04  (a, a);
BLK U05  (a, b);
BLK U06  (b, c);
BLK U07  (c, a);
BLK U08  (a, a);
BLK U09  (a, b);
BLK U10 (b, c);
BLK U11 (c, a);
BLK U12 (a, a);
BLK U13 (a, b);
BLK U14 (b, c);
BLK U15 (c, a);
BLK U16 (a, a);
BLK U17 (a, b);
BLK U18 (b, c);
BLK U19 (c, a);
BLK U20 (a, a);
 
BLK V01 (A, B);
BLK V02 (B, C);
BLK V03 (C, A);
BLK V04 (A, A);
BLK V05 (A, B);
BLK V06 (B, C);
BLK V07 (C, A);
BLK V08 (A, A);
BLK V09 (A, B);
BLK V10 (B, C);
BLK V11 (C, A);
BLK V12 (A, A);
BLK V13 (A, B);
BLK V14 (B, C);
BLK V15 (C, A);
BLK V16 (A, A);
BLK V17 (A, B);
BLK V18 (B, C);
BLK V19 (C, A);
BLK V20 (A, A);

endmodule


