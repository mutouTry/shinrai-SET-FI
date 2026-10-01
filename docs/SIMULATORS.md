# Simulators

`setfi record` runs the testbench on the gate-level netlist without timing
(no SDF, no timing checks). For each sampled cycle it records the value of
every net at the falling clock edge, after everything the testbench does at
that instant has settled. Cycle N is the cycle that begins at the N-th rising
clock edge with reset inactive (N = 1, 2, ...).

Requirements on the testbench:

* it instantiates the design as `workload.dut_instance` in `workload.tb_top`,
  with clock `workload.clock` and reset `workload.reset`;
* it does not change the design's inputs between a falling and the next rising
  clock edge (driving them at the falling edge, or just after the rising edge,
  is fine);
* it prints `workload.pass_string` (default `RESULT: PASS`) when it passes.

The simulation runs twice: first to count the cycles after reset and to check
the input timing, then to record the sampled ones into `record/states/`. For a
design without reset, set `workload.reset = ""`; cycles are then counted from
the first rising edge.

`` `include`` files are searched in the directories of the testbench files and
in `workload.include_dirs`. The files the testbench includes count as inputs
of `record`: editing one re-runs it.

## Icarus Verilog (default)

```toml
[simulator]
preset = "icarus"
```

Icarus has no register initialisation option, so storage that the design never
resets starts as X. It is recorded as 0, and cycles with more than
`workload.max_x_fraction` X/Z bits are replaced.

## VCS

```toml
[simulator]
preset = "vcs"
extra_args = ["-full64"]
# container = "vcs.sif"          # optional: run inside a container
# binds = ["/tools:/tools"]
```

With any preset, `env_unset` lists environment variables to remove before the
simulator runs, e.g. `["LD_LIBRARY_PATH"]` when a Python environment's
libraries disturb the simulator's link step.

## Other simulators

With `preset = "custom"`, give the compile and run commands as token lists
(`include_format` gives the form of an include-directory argument, default
`+incdir+{dir}`; `timescale` compiles a `` `timescale`` line first, for
simulators without a command-line default). The placeholders are:

| Placeholder | Meaning |
|---|---|
| `{filelist}` | file listing all sources, one per line |
| `{top}` | top module (the generated recording wrapper) |
| `{sim_bin}` | compiled simulation to write / run |
| `{work_dir}`, `{output_dir}` | working and output directories |
| `{defines}` | defines, each formatted with `define_format` |
| `{plusargs}` | recorder plusargs; pass them at run time |
| `{extra_args}`, `{run_args}` | from the same-named keys |

`recorder = "strobe"` samples with `$fstrobe` and works on any simulator;
`recorder = "program"` uses an SV `program` block.

## Adapting the testbench

The testbench is compiled with the gate-level netlist and the cell models
instead of the RTL. Most testbenches need nothing more. Three situations need
configuration:

* **The testbench refers to signals inside the design that synthesis renamed or
  flattened**, e.g. `dut.u_rf.mem[3]` when the netlist has a flat bus
  `u_rf_mem`. Rewrite the references with regex substitutions, applied to the
  testbench text in order:

  ```toml
  [workload]
  tb_substitutions = [
    { regex = 'dut\.u_rf\.mem\[(\w+)\]', replacement = 'dut.u_rf_mem[(7-\1)*16 +: 16]' },
  ]
  ```

  With `auto_derive_substitutions = true` and the design's RTL in `rtl_files`,
  such rewrites are derived from the RTL declarations for references that
  `tb_substitutions` does not cover; ambiguous references are reported.

* **The netlist instantiates macros without a model** (memories, IP), which
  `library.blackbox_cells` lets the build treat as black boxes. For the
  simulation, give their behavioural RTL:

  ```toml
  macro_rtl_substitutions = [{ macro_module = "sram_sp_1024x32", rtl_file = "rtl/sram.v" }]
  ```

  Netlist variants of a macro that synthesis uniquified with parameters
  (`sram_sp_1024x32_0`, ...) are mapped to the same RTL.

* **A module of the netlist must be simulated from other sources**: list it in
  `netlist_strip_modules` (it is removed from the copy of the netlist that is
  simulated) and give its replacement in `netlist_substitute_files`.

The adapted testbench and a report are written to `record/sim/testbench/`.
