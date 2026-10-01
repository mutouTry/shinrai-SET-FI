# Model

## Trials

A trial is a triple (cycle, site, width).

* **Cycle**: a clock cycle sampled from the workload. The circuit is in the
  state recorded for that cycle: the value of every net at the falling clock
  edge, after everything the testbench does at that instant has settled. As
  long as the design inputs do not change between the falling and the next
  rising edge, this is the state in which the rising edge samples the data
  pins.
* **Site**: a gate output between flip-flops: driven, through gates, by a
  flip-flop or black-box output, and able to reach a flip-flop data pin. Nets
  that drive a data pin are sites too (`fault_model.inject_ff_d_nets`), also
  when such a net is a primary input or another flip-flop's output wired
  straight to the data pin. Other primary inputs and flip-flop outputs,
  black-box outputs and logic driven only by primary inputs are not sites: the
  analysis covers the logic between the design's own flip-flops. The module
  hierarchy of the netlist does not matter.
* **Width** `w`: the SET inverts the site net at time 0 and restores it at
  time `w`.

The circuit state is held fixed during the trial, so the SET start time `s`
only shifts the result and is not part of the trial.

## Propagation

The pulse is simulated event by event through the fan-out cone of the site:

* each gate whose input changes is re-evaluated from its truth table
  (logical masking);
* an output transition takes the SDF `IOPATH` delay of the input that causes
  it, for the rise or fall at the output, using `COND` entries selected by the
  side inputs. If several inputs change at once, the largest of their delays
  is used;
* a scheduled transition is cancelled if its cause reverts before the delay
  has elapsed (inertial filtering), so a pulse narrower than a gate's delay
  dies at that gate;
* each net holds at most one pending transition;
* SDF `INTERCONNECT` delays can be added as pure transport delays.

Every excursion of a flip-flop data pin from its fault-free value is recorded
as a pulse `[t_enter, t_exit)` relative to the SET start.

## Capture

The SET start time `s` is uniform on `[0, T)` (1-ps resolution), and every
flip-flop samples at the capture edge `T`. The flip-flop's SDF setup and hold
checks define its sampling window. A pulse away from fault-free value `b` is
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
them, so the widths act as the points of a discrete distribution; inject evenly
spaced widths to approximate a continuous one.

## Scope

* One SET per trial, on one net; every site has the same weight.
* An error is a wrong value captured at a flip-flop data pin. Pulses that reach
  only clock, clock-enable or asynchronous set/reset pins are not counted.
* Every flip-flop captures at the same instant `T`; clock-tree delays in the
  SDF are not used.
* Only the SDF arcs inside the fan-out cones of the sites are used.
