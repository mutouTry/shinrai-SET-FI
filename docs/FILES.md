# Output files

Everything is written under `output.dir`, one directory per stage.

## stages.json

For each stage that has run: the configuration values and input-file checksums
it read, digests of the earlier stages' results it used, a digest of its own
result, and a short summary. `setfi status` and `setfi run` compare it with the
current configuration to decide what is out of date. Without it every stage
runs again.

## build/

The data `inject` and `logic` read. Nets and flip-flops are numbered here, and
the other stages refer to these numbers.

| File | Content |
|---|---|
| `net_index.json` | every net, numbered (`idx_to_net`, `net_to_idx`) |
| `ff_index.json` | `ffid_to_info` (the observed flip-flops: instance, cell, data pin, data net), `dnet_to_ffids` and `qnet_to_ffids` (data and output nets of every sequential cell), `_meta.observe_filter` (what `[observe]` removed) |
| `site_cones.jsonl` | one line per injection site: its net, the nets of its fan-out cone in topological order, the gate arcs between them, the flip-flop data nets it reaches |
| `edges.json` | one entry per gate arc: instance, input pin, output pin, nets |
| `pin_to_net.json` | instance pin to net |
| `cell_arcs.json` | per cell arc (every gate arc whose output depends on its input, with or without an SDF delay): output function, side inputs, truth table, side-input values that let a change pass |
| `timing_checks.json` | per flip-flop: its SDF setup and hold (and recovery, removal) checks |
| `delay_table.npz`, `arc_index.json` | rise and fall delay of every instance arc, with its SDF `COND` alternatives. `arc_index.json` gives each arc's row |
| `interconnect.json` | only if the SDF has `INTERCONNECT`: wire delay per load pin |
| `cell_functions.v` | only for Liberty (or several) `library.cell_functions`: the cells as Verilog modules |
| `build_manifest.json` | input checksums, counts, check results, and SDF entries that are not used, by reason. With `build.allow_missing_delays`, `arcs_without_delay` counts the gate arcs inside the fan-out cones that have no SDF delay (only if there are any) |
| `intermediate/` | only with `build.keep_intermediates` |

## record/

| File | Content |
|---|---|
| `states/cycle_<N>.hex` | one hexadecimal number: bit `i` is the value of net `i` of `build/net_index.json` at the falling clock edge of cycle `N` (cycle `N` begins at the `N`-th rising edge after reset). X or Z digits are read as 0. |
| `cycles.json` | `cycles` (the sampled cycles), how they were sampled, and how many unknown values they hold |
| `record.log`, `result.json` | the recording's log, and its status and simulator commands |
| `sim/` | simulator working files: generated wrapper, file lists, logs, `testbench/` (report of the testbench adaptation, and the adapted testbench when substitutions apply). Can be deleted. |

## inject/

`index.json` (format `setfi-records/2`) describes the campaign:
`clock_period_ps`, `pulse_widths_ps`, `cycles`, `sites` (site index to net
name), `ffs` (flip-flop id, instance, data net, setup and hold,
`capture_enter_ps` and `capture_leave_ps` in most cycles), `windows` (the file
`windows.npz`), `shards` (the record files), `propagation` (settings used) and
`counts`. `counts.arcs_without_delay` is there only if the sites' cones hold
gate arcs without a delay, which pulses do not pass. `windows.npz` lists the
cycles in which a flip-flop's window differs:
`cycle`, `ff`, and `enter`, `leave` of shape [n, 2] by `base`.

`records_c<i>_s<j>.npz` hold the pulse records of block `i` of the sampled
cycles (`inject.cycles_per_shard` cycles each) and slice `j` of the sites (one
slice per parallel job). One row is one pulse seen at the data pin of one
observed flip-flop in one trial (cycle, site, width):

| Column | Type | Meaning |
|---|---|---|
| `cycle` | int32 | sampled cycle |
| `site` | int32 | index into `sites` |
| `width_ps` | int32 | injected pulse width |
| `ff` | int32 | flip-flop id |
| `base` | uint8 | fault-free value of the data pin |
| `t_enter`, `t_exit` | int64 | start and end of the pulse at the data pin, in ps after the SET start. `t_exit` = 2^62 if the pulse lasts beyond the simulated time |

`capture_enter_ps` and `capture_leave_ps` have two entries, for `base` 0 and 1:
`enter = T + hold` and `leave = T - setup` for the data edges that start and end
the pulse. The flip-flop captures a wrong value for a SET starting at `s` (ps
from the start of the cycle) if

```
leave[base] - t_exit + 1  <=  s  <=  enter[base] - t_enter - 1
```

Trials without a row reach no observed flip-flop. `src/setfi/records.py` reads
the format.

## patterns/

Only with `[patterns] steps` > 0. The SET start times are the grid
`starts_ps[k]` = k*T/steps for k = 0..steps, rounded half up to integer ps, where
T is `capture_edge_ps`, the clock period. At `starts_ps[k]` a trial (cycle, site,
width) upsets the set F of flip-flops that one of its records makes capture (the
rule above). A pattern is a pair (cycle, F) with F not empty. Grid points at
which a trial upsets no flip-flop belong to no pattern.

`patterns.npz` (format `setfi-patterns/1`, numpy arrays only):

| Array | Type | Meaning |
|---|---|---|
| `format` | str | `setfi-patterns/1` |
| `period_ps`, `steps` | int64 | T and the number of grid steps |
| `starts_ps` | int64[steps + 1] | the grid of start times |
| `cycles`, `pulse_widths_ps` | int64 | the sampled cycles and the injected widths of the campaign |
| `sites` | str | the injection sites, indexed by `run_site` |
| `ff_ids`, `ff_instances` | int32, str | the observed flip-flops: id and instance name |
| `pattern_cycle` | int32[P] | the cycle of pattern p |
| `pattern_ff_ptr`, `pattern_ffs` | int64[P + 1], int32 | the flip-flop ids of pattern p, ascending: `pattern_ffs[pattern_ff_ptr[p]:pattern_ff_ptr[p + 1]]` |
| `pattern_multiplicity` | int64[P] | the number of grid points (site, width, k) of the cycle at which a SET upsets exactly F |
| `pattern_n_sites` | int32[P] | the number of sites among them |
| `pattern_run_ptr` | int64[P + 1] | the runs of pattern p: rows `pattern_run_ptr[p]` to `pattern_run_ptr[p + 1] - 1` of the `run_` arrays |
| `run_site`, `run_width_ps` | int32[R] | the trial of a run: site index and pulse width |
| `run_k_lo`, `run_k_hi` | int32[R] | first and last grid index of the run: a maximal range of k at which the trial upsets exactly F |

Patterns are sorted by cycle, then by their flip-flop ids (compared as tuples).
The runs of a pattern are sorted by site, width and `run_k_lo`. The run lengths of a
pattern add up to its multiplicity. `patterns.csv` has one row per pattern,
`pattern,cycle,multiplicity,n_ffs,n_sites,ffs` (`pattern` is its index p in
`patterns.npz`, `ffs` its flip-flop ids separated by `|`). `src/setfi/analysis/patterns.py`
reads the format.

## logic/

Written by `setfi logic`: `patterns.npz` and `patterns.csv` in the format of
`patterns/`, which `src/setfi/analysis/patterns.py` reads unchanged. A pattern
is a pair (cycle, F), F not empty: the flip-flops whose data pins change when a
site is inverted in that cycle (docs/MODEL.md). `cycles`, `sites`, `ff_ids` and
`ff_instances` are those `inject` uses with the same configuration. The table
has no start times and no pulse widths, and holds neutral values for them:

| Array | Value |
|---|---|
| `period_ps`, `steps` | 0 (a table of `patterns/` has steps >= 1) |
| `starts_ps` | `[0]` |
| `pulse_widths_ps` | empty |
| `run_width_ps`, `run_k_lo`, `run_k_hi` | 0 |

Each run is one site, so `pattern_multiplicity` and `pattern_n_sites` both
count the sites that give the pattern, and `Patterns.grid_points()` has one row
per (cycle, site) whose SET changes a data pin.

## analyze/\<name\>/

| File | Content |
|---|---|
| `summary.json` | probability that a SET upsets at least one flip-flop, the same per width, the distribution of the number of flip-flops upset together, total exposure, and the width model and definitions |
| `per_ff.csv` | probability that a SET upsets each flip-flop |
| `per_site.csv` | probability that a SET at a site upsets a flip-flop |
| `per_width.csv` | upset probability for each injected width |
| `trial_exposure.npz` | only with `analyze.write_trials`: one row per trial that can upset a flip-flop (`cycle`, `site` as an index into `site_names`, `width_ps`, `exposure_ps` > 0). All other trials have exposure 0 |

The directory is named after the width model (`uniform`,
`gaussian_mean90_sd40`, `table_<hash>`, with the suffix `_closed` for the
closed phase domain) or by `--name`.
