# Output files

Everything is written under `output.dir`, one directory per stage.

## stages.json

For each stage that has run: the configuration values and input-file checksums
it read, digests of the earlier stages' results it used, a digest of its own
result, and a short summary. `setfi status` and `setfi run` compare it with the
current configuration to decide what is out of date. Without it every stage
runs again.

## build/

The data `inject` reads. Nets and flip-flops are numbered here; the other
stages refer to these numbers.

| File | Content |
|---|---|
| `net_index.json` | every net, numbered (`idx_to_net`, `net_to_idx`) |
| `ff_index.json` | `ffid_to_info`: the observed flip-flops (instance, cell, data pin, data net); `dnet_to_ffids`, `qnet_to_ffids`: data and output nets of every sequential cell; `_meta.observe_filter`: what `[observe]` removed |
| `site_cones.jsonl` | one line per injection site: its net, the nets of its fan-out cone in topological order, the gate arcs between them, the flip-flop data nets it reaches |
| `edges.json` | one entry per gate arc: instance, input pin, output pin, nets |
| `pin_to_net.json` | instance pin to net |
| `cell_arcs.json` | per cell arc: output function, side inputs, truth table, side-input values that let a change pass |
| `timing_checks.json` | per flip-flop: its SDF setup and hold (and recovery, removal) checks |
| `delay_table.npz`, `arc_index.json` | rise and fall delay of every instance arc, with its SDF `COND` alternatives; `arc_index.json` gives each arc's row |
| `interconnect.json` | only if the SDF has `INTERCONNECT`: wire delay per load pin |
| `cell_functions.v` | only for Liberty (or several) `library.cell_functions`: the cells as Verilog modules |
| `build_manifest.json` | input checksums, counts, check results, and SDF entries that are not used, by reason |
| `intermediate/` | only with `build.keep_intermediates` |

## record/

| File | Content |
|---|---|
| `states/cycle_<N>.hex` | one hexadecimal number: bit `i` is the value of net `i` of `build/net_index.json` at the falling clock edge of cycle `N` (cycle `N` begins at the `N`-th rising edge after reset). X or Z digits are read as 0. |
| `cycles.json` | `cycles`: the sampled cycles; how they were sampled; how many unknown values they hold |
| `record.log`, `result.json` | the recording's log; its status and simulator commands |
| `sim/` | simulator working files: generated wrapper, file lists, logs, `testbench/` (report of the testbench adaptation, and the adapted testbench when substitutions apply). Can be deleted. |

## inject/

`index.json` (format `setfi-records/1`) describes the campaign:
`clock_period_ps`, `pulse_widths_ps`, `cycles`, `sites` (site index to net
name), `ffs` (flip-flop id, instance, data net, setup and hold,
`capture_enter_ps` and `capture_leave_ps`), `shards` (the record files),
`propagation` (settings used) and `counts`.

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
| `t_enter`, `t_exit` | int64 | start and end of the pulse at the data pin, ps after the SET start; `t_exit` = 2^62 if the pulse lasts beyond the simulated time |

`capture_enter_ps` and `capture_leave_ps` have two entries, for `base` 0 and 1:
`enter = T + hold` and `leave = T - setup` for the data edges that start and end
the pulse. The flip-flop captures a wrong value for a SET starting at `s` (ps
from the start of the cycle) if

```
leave[base] - t_exit + 1  <=  s  <=  enter[base] - t_enter - 1
```

Trials without a row reach no observed flip-flop. `src/setfi/records.py` reads
the format.

`phase_grid_captures.csv` (only with `inject.phase_grid_steps`): for the start
times `k*T/steps`, one row `cycle,site,start_ps,width_ps,ffs` per SET that
upsets flip-flops; `ffs` lists their ids, separated by `|`.

## analyze/\<name\>/

| File | Content |
|---|---|
| `summary.json` | probability that a SET upsets at least one flip-flop; the same per width; distribution of the number of flip-flops upset together; total exposure; the width model and definitions |
| `per_ff.csv` | probability that a SET upsets each flip-flop |
| `per_site.csv` | probability that a SET at a site upsets a flip-flop |
| `per_width.csv` | upset probability for each injected width |
| `trial_exposure.npz` | only with `analyze.write_trials`: one row per trial that can upset a flip-flop (`cycle`, `site` as an index into `site_names`, `width_ps`, `exposure_ps` > 0); all other trials have exposure 0 |

The directory is named after the width model (`uniform`,
`gaussian_mean90_sd40`, `table_<hash>`; suffix `_closed` for the closed phase
domain) or by `--name`.
