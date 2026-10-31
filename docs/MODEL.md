# Model

setfi asks, for many trials: if one net is inverted for `w` ps in a given clock
cycle, which flip-flops capture a wrong value, and for which SET start times?
This page defines a trial, how its pulse propagates, when a flip-flop captures
it, and how trials are combined into probabilities.

## Trials

A trial is a triple (cycle, site, width).

* **Cycle**: a clock cycle sampled from the workload. The circuit is in the
  state recorded for that cycle: the value of every net at the falling clock
  edge, after everything the testbench does at that instant has settled. As
  long as the design inputs do not change between the falling and the next
  rising edge, this is the state in which the rising edge samples the data
  pins.
* **Site**: a gate output between flip-flops: driven, through gates, by a
  flip-flop or black-box output, and able to reach a flip-flop data pin. With
  `fault_model.inject_ff_d_nets`, nets that drive a data pin are sites too,
  also when such a net is a primary input or another flip-flop's output wired
  straight to the data pin. Other primary inputs and flip-flop outputs,
  black-box outputs and logic driven only by primary inputs are not sites. The
  analysis covers the logic between the design's own flip-flops. Sites are the
  same whether the netlist is hierarchical or flat.
* **Width** `w`: the SET inverts the site net at time 0 and restores it at
  time `w`.

The circuit state is held fixed during the trial, so the SET start time `s`
only shifts the result and is not part of the trial.

## Propagation

The pulse is simulated event by event through the fan-out cone of the site.
Every excursion of a flip-flop data pin from its fault-free value is recorded
as a pulse `[t_enter, t_exit)` relative to the SET start.

### Logical masking

Each gate whose input changes is re-evaluated from its truth table. A change
that the side inputs block goes no further.

### Delays

An output transition takes the SDF `IOPATH` delay of an input that changed,
for the rise or fall at the output. Where the SDF gives `COND` entries, it is
the delay of the entry that holds for the side-input values after the change.
If none holds (and the side-input values are known), the input's path is
inactive. The default delay of an arc is its unconditional SDF value or, if
there is none, the median of its conditional ones.

A single changed input gets the delay of its `COND` entry that holds, or else
its default delay. When several inputs change at the same instant, every
changed input with an active path competes and the smallest delay is used, as
for module paths in a Verilog simulator. If none of them has an active path,
the inputs that alone explain the output change (all changed inputs if none
does) compete with their default delays, and `propagation.delay_choice` picks
the largest (default) or the smallest.

An arc without an SDF delay passes no pulse: a change of its input does not
reach the output through it. The build accepts such arcs inside the fan-out
cones only with `build.allow_missing_delays`, and `build_manifest.json` and
`inject` count them. The enable arc of a three-state output may be given only
by its transitions to and from Z; [CELLS.md](CELLS.md#three-state-outputs)
gives the rise and fall it takes from them.

### Inertial filtering

Each net holds at most one pending transition. If an input changes before the
transition happens and the re-evaluated gate gives the net's current value
again, the transition is cancelled. A pulse narrower than a gate's delay dies
at that gate, while a pulse whose new value another input still drives passes.
A re-evaluation that gives the pending value again keeps the transition at its
scheduled time.

### Wire delays

SDF `INTERCONNECT` delays can be added as pure transport delays
(`propagation.interconnect`).

## Capture

The SET start time `s` is uniform on `[0, T)` (1-ps resolution), and every
flip-flop samples at the capture edge `T`. The flip-flop's SDF setup and hold
checks define its sampling window. Where the SDF gives a check several entries
with a `COND`, a cycle uses the largest of those whose `COND` holds in the
recorded state, or 0 ps if none holds. A pulse away from fault-free value `b` is
captured, and the flip-flop latches a wrong value, when the pulse overlaps the
window:

```
leave_b - t_exit + 1  <=  s  <=  enter_b - t_enter - 1
enter_b = T + hold(leading edge),   leave_b = T - setup(trailing edge)
```

For trial `k`, the **capture region** `A_kf` of flip-flop `f` is the union of
these start-time intervals over its pulses. The **exposure** `Omega_k` is
the union of `A_kf` over all flip-flops: the start times at which the SET
upsets at least one flip-flop.

## Results

Sites and cycles are weighted uniformly, start times uniformly over `[0, T)`,
and widths by the width model `p(w)`.

| Result | Definition |
|---|---|
| upset probability | `sum_k p(w_k) * len(Omega_k) / (N_cycles * N_sites * T)` |
| per flip-flop | the same with `len(A_kf)` |
| per site | the same, restricted to one site |
| per width | conditional on each width |
| multi-bit upsets | probability that exactly `m` flip-flops capture together |

The width model gives each injected width a probability: `uniform`,
`gaussian` (`mean_ps`, `sd_ps`), `exponential` (`tau_ps`) or `table` (explicit
weights). The density is evaluated at the injected widths and normalised over
them, so the widths act as the points of a discrete distribution. Inject evenly
spaced widths to approximate a continuous one.

## Logic only

`setfi logic` keeps the cycles, the sites and logical masking, and drops the
width, the start time and all delays. For each (cycle, site), the site is
inverted in the recorded state and the gates of its fan-out cone whose inputs
change are re-evaluated in topological order. Every other net keeps its
recorded value. Every observed flip-flop whose data pin then differs from its
recorded value is upset. There is no inertial filtering and no capture window,
and paths that reconverge give the steady value, not a glitch. The result is
the set F of upset flip-flops per (cycle, site).

## Clock gates

By default (`observe.clock_gates = "ignore"`) an integrated clock gate is not an
observation point and every flip-flop gets the fault-free clock edge. A SET that
changes what the gate latches is then not in F, and a flip-flop whose gate is
off in the fault-free run can still be in F.

With `"expand"`, the gates of the Liberty types `latch_posedge*` are observed
through their enable pin, with the setup and hold of that pin against the gate's
clock. A pulse changes the latched value when it overlaps that window. A pulse
that ends earlier, while the latch is still transparent, leaves no trace. F then
holds gates as well as flip-flops, and the `patterns` and `logic` stages rewrite
each (cycle, F) into (cycle, F'), which holds flip-flops only.

For each flip-flop, E0 and E1 tell whether it gets an edge at `T` without and
with the SET. Each is the product of the enables on its clock path, with an
enable inverted if F holds that gate. An enable is the enable pin ORed with the
test-enable pin, so a flipped enable has no effect while test enable is 1. D and
Q are read from the recorded state of the cycle.

| E0 | E1 | The flip-flop is in F' when | |
|---|---|---|---|
| 1 | 1 | it is in F | the normal case |
| 0 | 0 | not in F' | it has no edge, also if it is in F |
| 0 | 1 | (D XOR it is in F) differs from Q | an extra edge captures D under the SET |
| 1 | 0 | D differs from Q | the edge is lost, so a SET on D has no effect |

Gates in series follow from the product: a flip upstream changes the groups
behind the gates whose own enable is 1. After `T` nothing of the SET is left,
because the latch is transparent again in the low phase of the next cycle.

Grid points whose F' is empty belong to no pattern, and the report counts them.
Equal (cycle, F') are merged with their runs joined, in the usual table format.
`analyze` works from the pulse records and leaves the gates out, so its
results are those of `"ignore"` and do not include the flip-flops that only a
clock-edge change flips.

The rewrite was compared with an SDF-annotated gate-level simulation on one small
processor netlist, at 1,300 trials. Gates in series are covered by unit tests
only, since that netlist has none.

What F' does not express is left as it is and reported, per pattern in
`clock_gates.json` of the stage and per design in `build/clock_gates.json`:

* A gate with a memory or another black box in its clock fan-out. An edge more or
  less is a memory operation more or less, not a flip-flop flip.
* An enable, D or Q recorded as X or Z. It is read as 0, like elsewhere in setfi.
* Cells behind a gate that do not sample its rising edge (latches, the falling
  edge), and cells behind several gates. Their entries in F stay.
* A flip-flop in F' from a gate whose asynchronous set or reset is active in the
  recorded state, and a flip-flop that would flip but is not observed.
* Gates of other Liberty types, and other sequential cells that drive a clock
  pin, such as a latch ANDed with the clock or a register-generated clock. They
  are counted by the build. The flip-flops behind them are treated as clocked.

F' can hold flip-flops that an injection tool cannot reach, for example those
outside a scan chain. setfi does not know the chains, so that is for the tool
that reads the table. In the table above, D stands for the value the edge would
load, from the cell's model: the data pin through a scan mux or a synchronous
clear, if the cell has one. Entries of F still come from the data pin alone.

## Scope

* One SET per trial, on one net. Every site has the same weight.
* An error is a wrong value captured at a flip-flop data pin. Pulses that reach
  only clock, clock-enable or asynchronous set/reset pins are not counted, apart
  from integrated clock gates with `observe.clock_gates = "expand"`.
* Every flip-flop captures at the same instant `T`. Clock-tree delays in the
  SDF are not used.
* Only the SDF arcs inside the fan-out cones of the sites are used.
