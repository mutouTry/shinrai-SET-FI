# setfi

Timing-aware analysis of single-event transients (SETs) in gate-level designs.

setfi injects SET pulses at every gate output that can reach a flip-flop, in
clock cycles sampled from a workload. It propagates each pulse through the
netlist with the SDF delays and records where it arrives at flip-flop data
pins. From these records it computes, for every SET start time in the cycle,
which flip-flops capture a wrong value.

## Flow

| Stage | Reads | Writes |
|---|---|---|
| `build` | netlist, SDF, cell functions | `build/`: timing graph, injection sites, fan-out cones |
| `record` | `build/`, testbench, simulator | `record/`: circuit state of the sampled cycles |
| `inject` | `build/`, `record/`, pulse widths | `inject/`: one record per pulse arriving at a flip-flop data pin |
| `analyze` | `inject/`, width models | `analyze/<width model>/`: upset probabilities |

Propagation accounts for logical masking, inertial (electrical) filtering,
rise/fall and conditional delays, and reconvergent paths. One simulation per
(cycle, site, width) covers all start times in the cycle, so any pulse-width
distribution can be evaluated afterwards without a new campaign. The model is
defined in [docs/MODEL.md](docs/MODEL.md).

## Install

```sh
pip install .
```

Requires Python 3.9+, numpy and a C++17 compiler; pip fetches pybind11 for the
build (offline: install `pybind11` first and use `pip install --no-build-isolation .`).
The `record` stage needs Icarus Verilog, VCS or another Verilog simulator
([docs/SIMULATORS.md](docs/SIMULATORS.md)). Tests: `pip install ".[test]" && pytest`.

## Inputs

* **Netlist**: structural Verilog with named port connections. Modules may be
  hierarchical; `assign` statements between nets are treated as connections.
  Escaped identifiers are not supported.
* **SDF** for that netlist: an `IOPATH` delay (with or without `COND`) for every
  gate arc a pulse can traverse, `SETUP` and `HOLD` (or `SETUPHOLD`) checks for
  every observed flip-flop, and optionally `INTERCONNECT`.
* **Cell functions**: a Liberty file, or Verilog cell modules
  ([docs/CELLS.md](docs/CELLS.md)).
* **Testbench**: instantiates the design, runs it with a clock (and usually a
  reset), and prints a pass string. It must not change the design's inputs
  between a falling and the next rising clock edge; `record` checks this.

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

Each stage can be run alone: `setfi build | record | inject | analyze -c setfi.toml`.
A stage runs only if a configuration key it reads (the template says which),
one of its input files, or the result of an earlier stage has changed
(`--force` runs it anyway). Width models only weight the pulse records, so
another one needs only `analyze`:

```sh
setfi analyze -c setfi.toml --width-model '{"type": "gaussian", "mean_ps": 60, "sd_ps": 20}' --name narrow
```

Sampling is reproducible: with the same seed, increasing `workload.n_cycles`
keeps the cycles of the smaller sample and adds new ones.

## Output directory

```
<dir>/
  stages.json     what each stage ran from; decides what is out of date
  build/          timing graph, injection sites and fan-out cones, read by inject
  record/         states/cycle_<N>.hex: the circuit state of each sampled cycle
  inject/         pulse records
  analyze/
    <model>/      summary.json, per_ff.csv, per_site.csv, per_width.csv
```

Every file is described in [docs/FILES.md](docs/FILES.md).

`analyze` reads only `inject/`. After a campaign, `build/` and `record/` can be
deleted to save space: the pulse records stay valid, and a later `setfi run`
re-creates `build/` and `record/` without redoing the campaign. When an input has
changed since the campaign, `analyze` says so. A campaign whose pulse records
differ from the previous ones deletes `analyze/`.

One output directory holds one campaign: changing a key that `inject` reads
replaces the campaign (`setfi status` shows what a run would redo). To compare
campaign settings (pulse widths, clock period, propagation options), copy the
output directory, point a second configuration at the copy and change it there:
stages the change does not affect are reused. Width models need no new campaign.

## License

MIT
