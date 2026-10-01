# Cell descriptions

`library.cell_functions` gives the logic of every cell used in the netlist, as
Liberty files, Verilog cell modules, or both.

## Liberty

Liberty files (`.lib`) are translated automatically. The translation uses each
output pin's `function` and the cell's `ff`, `latch` or
`clock_gating_integrated_cell` description. Three-state outputs, buses and
`statetable` cells are not translated; the build stops with an error if the
netlist uses one of them.

## Verilog

One module per cell, written with continuous assignments and `always` blocks
(not gate primitives or UDPs):

```verilog
module NAND2 (input A, input B, output Z);
  assign Z = !(A & B);
endmodule

module AOI21 (A1, A2, B, ZN);
  input A1, A2, B;
  output ZN;
  wire n;
  assign n = A1 & A2;          // intermediate wires are allowed
  assign ZN = ~(n | B);
endmodule

module DFFR (input D, input CK, input RN, output reg Q, output QN);
  always @(posedge CK or negedge RN)
    if (!RN) Q <= 1'b0;
    else Q <= D;
  assign QN = !Q;
endmodule

module LATCH (input D, input G, output reg Q);
  always @(*) if (G) Q <= D;
endmodule
```

* Combinational outputs: `assign` with `~ ! & | ^ ?:`, parentheses and the
  constants `1'b0` / `1'b1`.
* Sequential cells are recognised by a state output, a data input and a
  clock. The clock is the signal in `always @(posedge ...)` or
  `always @(negedge ...)`; for latches it is taken from
  `library.clock_pins`. Output, data and asynchronous pins are recognised by
  name (`library.q_pins`, `library.d_pins`, `library.async_pins`).
* Sequential cells end the combinational logic: a pulse is observed at their
  data pin and does not propagate through them.

The delays come from the SDF, never from the cell descriptions.
