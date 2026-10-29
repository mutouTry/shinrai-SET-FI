# setfi

Timing-aware analysis of single-event transients (SETs) in gate-level designs.

setfi injects SET pulses at the gate outputs between flip-flops, in clock
cycles sampled from a workload. It propagates each pulse through the netlist
with the SDF delays and records where it arrives at flip-flop data pins. From
these records it computes, for every SET start time in the cycle, which
flip-flops capture a wrong value.

## Flow

| Stage | Reads | Writes |
|---|---|---|
| `build` | netlist, SDF, cell functions | `build/`: timing graph, injection sites, fan-out cones |
| `record` | `build/`, testbench, simulator | `record/`: circuit state of the sampled cycles |
| `inject` | `build/`, `record/`, pulse widths | `inject/`: one record per pulse arriving at a flip-flop data pin |
| `patterns` | `inject/` | `patterns/`: the sets of flip-flops upset together, and where from (optional) |
| `analyze` | `inject/`, width models | `analyze/<width model>/`: upset probabilities |

Propagation accounts for logical masking, inertial (electrical) filtering,
rise/fall and conditional delays, and reconvergent paths. One simulation per
(cycle, site, width) covers all start times in the cycle, so a distribution over
the injected pulse widths can be evaluated afterwards without a new campaign.
The model is defined in [docs/MODEL.md](docs/MODEL.md).

## Install

```sh
pip install .
```

Requires Python 3.9+, numpy and a C++17 compiler. The build fetches pybind11
(offline: install `pybind11` first and use `pip install --no-build-isolation .`).
The `record` stage needs Icarus Verilog, VCS or another Verilog simulator
([docs/SIMULATORS.md](docs/SIMULATORS.md)). Tests: `pip install ".[test]" && pytest`.

## Inputs

* **Netlist**: structural Verilog with named port connections. Modules may be
  hierarchical, and `assign` statements between nets are treated as connections.
  Escaped identifiers are not supported.
* **SDF** for that netlist: an `IOPATH` delay (with or without `COND`) for every
  gate arc a pulse can traverse (for the enable of a three-state output, its
  transitions to and from Z suffice: [docs/CELLS.md](docs/CELLS.md)), `SETUP`
  and `HOLD` (or `SETUPHOLD`) checks for every observed flip-flop, and
  optionally `INTERCONNECT`.
* **Cell functions**: a Liberty file, or Verilog cell modules
  ([docs/CELLS.md](docs/CELLS.md)).
* **Testbench**: instantiates the design, runs it with a clock (and usually a
  reset), and prints a pass string. It must not change the design's inputs
  between a falling and the next rising clock edge. `record` checks this.

## Usage

```sh
setfi init setfi.toml       # template: every option, its default and which stage uses it
setfi config -c setfi.toml  # the configuration with defaults filled in, validated
setfi run -c setfi.toml     # every stage that is out of date, then analyze
setfi status -c setfi.toml  # which stages are out of date, and why
```

A minimal configuration:

```toml
[design]
netlist = "mapped.v"
sdf = "mapped.sdf"
top = "my_top"
clock_period_ns = 1.0

[output]
dir = "runs/my_top"

[library]
cell_functions = ["cells.lib"]

[workload]
testbench = ["tb.sv"]
tb_top = "tb"                 # module, clock, reset and instance names in the testbench
dut_instance = "dut"
clock = "clk"
reset = "rst_n"
n_cycles = 200

[analyze]
width_models = [{ type = "gaussian", mean_ps = 90, sd_ps = 40 },
                { type = "exponential", tau_ps = 40 }]
```

Each stage can also be run alone: `setfi build | record | inject | patterns | analyze -c setfi.toml`.
`setfi logic` writes the pattern table without timing ([docs/MODEL.md](docs/MODEL.md#logic-only)).

A stage runs only when it is out of date: a configuration key it reads (the
template says which), one of its input files, or the result of an earlier stage
has changed. `--force` runs it anyway. Width models only weight the pulse
records, so trying another one needs only `analyze`:

```sh
setfi analyze -c setfi.toml --width-model '{"type": "gaussian", "mean_ps": 60, "sd_ps": 20}' --name narrow
```

Sampling is reproducible: with the same seed, increasing `workload.n_cycles`
keeps the cycles of the smaller sample and adds new ones.

## Output directory

```
<dir>/
  stages.json     what each stage ran from; decides what is out of date
  build/          timing graph, injection sites and fan-out cones, read by inject and logic
  record/         states/cycle_<N>.hex: the circuit state of each sampled cycle
  inject/         pulse records
  patterns/       patterns.npz, patterns.csv (with [patterns] steps > 0)
  logic/          patterns.npz, patterns.csv (setfi logic)
  analyze/
    <model>/      summary.json, per_ff.csv, per_site.csv, per_width.csv
```

`analyze` reads only `inject/`. After a campaign, `build/` and `record/` can be
deleted to save space. The pulse records stay valid, and a later `setfi run`
re-creates `build/` and `record/` without redoing the campaign. When an input
has changed since the campaign, `analyze` says so. A campaign whose pulse
records differ from the previous ones deletes `analyze/` and `patterns/`.

One output directory holds one campaign. Changing a key that `inject` reads
replaces the campaign (`setfi status` shows what a run would redo). To compare
campaign settings (pulse widths, clock period, propagation options), copy the
output directory, point a second configuration at the copy and change it there.
Stages the change does not affect are reused.

## Results

| File | Content |
|---|---|
| `analyze/<model>/` | Upset probabilities for one pulse-width model: `summary.json`, `per_ff.csv`, `per_site.csv`, `per_width.csv`. |
| `inject/index.json`, `inject/records_*.npz` | The pulse records: one row per pulse that reaches the data pin of an observed flip-flop in a trial (cycle, site, width). `patterns` and `analyze` read only these. |
| `patterns/patterns.npz`, `patterns/patterns.csv` | The pattern table. For SET start times on the grid k*T/steps (k = 0..`[patterns] steps`), each set F of flip-flops that SETs upset together in a cycle, with the grid points (site, width, k) at which a SET upsets exactly F. |
| `logic/patterns.npz`, `logic/patterns.csv` | The pattern table of `setfi logic`, in the same format: each set F of flip-flops whose data pins change together when a site is inverted in a cycle, with the sites that give it. |

```python
from setfi.analysis.patterns import Patterns
pt = Patterns.load("runs/my_top/patterns/patterns.npz")      # or logic/patterns.npz
print(pt.count, pt.cycle(0), pt.multiplicity(0), pt.ffs(0, names=True), pt.sources(0)[:3])
points = pt.grid_points()                                     # columns, one row per grid point
```

[docs/FILES.md](docs/FILES.md) gives the format of every file.

## License

MIT
