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

## Scope

* One SET per trial, on one net. Every site has the same weight.
* An error is a wrong value captured at a flip-flop data pin. Pulses that reach
  only clock, clock-enable or asynchronous set/reset pins are not counted.
* Every flip-flop captures at the same instant `T`. Clock-tree delays in the
  SDF are not used.
* Only the SDF arcs inside the fan-out cones of the sites are used.
