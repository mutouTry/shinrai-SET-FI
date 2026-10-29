# Cell descriptions

`library.cell_functions` gives the logic of every cell used in the netlist, as
Liberty files, Verilog cell modules, or both.

## Liberty

Liberty files (`.lib`) are translated automatically. The translation uses each
output pin's `function` and the cell's `ff`, `latch` or
`clock_gating_integrated_cell` description. Three-state outputs, buses and
`statetable` cells are not translated, and the build stops with an error if
the netlist uses one of them. A three-state cell can be given as a Verilog
module next to the Liberty file ([Three-state outputs](#three-state-outputs)).

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
  `always @(negedge ...)`. For latches it is taken from
  `library.clock_pins`. Output, data and asynchronous pins are recognised by
  name (`library.q_pins`, `library.d_pins`, `library.async_pins`).
* Sequential cells end the combinational logic: a pulse is observed at their
  data pin and does not propagate through them.

The delays come from the SDF, not from the cell descriptions.

## Three-state outputs

Describe a three-state output by the value of its net, also with the driver
off. For a net with a pull-up:

```verilog
module TBUF (input I, input OE, output Z);
  assign Z = OE ? I : 1'b1;   // driver off: the pull-up gives 1
endmodule
```

SDF writers often give the enable arc only by its transitions to and from Z:
`(IOPATH OE Z () () (0Z) (Z1) (1Z) (Z0))`, or `(IOPATH OE Z () () (Z))` with
one value for 0Z and 1Z. The arc's rise and fall then come from these values.
The enable turns the driver off at the value of its pin for which the
function no longer depends on the other inputs. The output is then a
constant p, the pulled value (here OE = 0 and p = 1). Through the enable the
output only moves between p and the driven value 1 - p:

| Output change | Driver | SDF value, p = 1 | SDF value, p = 0 |
|---|---|---|---|
| to p | turns off | 0Z | 1Z |
| to 1 - p | turns on | Z0 | Z1 |

A Verilog simulator times the net the same way: the driver goes to Z after the
turn-off delay, and the pull gives the net its value at that instant. If no
value of the enable pin makes the output constant, or the SDF gives neither
needed value, the arc has no delay (`build.allow_missing_delays`).
