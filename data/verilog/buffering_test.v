module TOP ( );
wire a,b,c;
wire x,y,z;
  
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
 
BLK V01 (x, y);
BLK V02 (y, z);
BLK V03 (z, x);
BLK V04 (x, x);
BLK V05 (x, y);
BLK V06 (y, z);
BLK V07 (z, x);
BLK V08 (x, x);
BLK V09 (x, y);
BLK V10 (y, z);
BLK V11 (z, x);
BLK V12 (x, x);
BLK V13 (x, y);
BLK V14 (y, z);
BLK V15 (z, x);
BLK V16 (x, x);
BLK V17 (x, y);
BLK V18 (y, z);
BLK V19 (z, x);
BLK V20 (x, x);

endmodule


