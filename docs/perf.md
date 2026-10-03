# M1 performance record

Machine: Apple M4, 10 cores (4 performance + 6 efficiency), 24 GB. Release build (`opt-level 3`, fat LTO, 1 codegen unit).
Date: 2026-10-04. Code commit for every measurement: `7eb3732` (`feat(bench): per-phase tick timing`). No optimization commits followed it (see "Decision").

Commands use `abi-bench` defaults unless shown: seed 6, `--size 256`, `--graze 100`, 8 threads for `throughput` and `phases`.

## Budget table

| metric | budget | measured | result |
|---|---|---|---|
| us_per_agent_step, 8 threads, 20k run | <= 0.4 | 0.811 | FAIL |
| speedup, 8 threads vs 1 | >= 6.0 | 2.85 | FAIL |
| bytes_per_agent, tick-0 founders (hidden 4..13) | <= 2048 | 1318 | pass |
| bytes_per_agent, evolved population (128x128, 2000 founders, 3000 ticks warmup; hidden-24 genome alone is ~2.35 KB) | <= 2048 | 1404 | pass |
| checkpoint_bytes, 256x256, 20k agents | <= 100 MB | 14,131,554 | pass |
| ticks_per_s, 256x256, 100k founders falling to ~16k (not the spec's sustained 100k on 512x512; at ~55k agents the phase run gives ~41 ms/tick ≈ 24 t/s) | >= 60 | 33.6 | FAIL |

All M1 budgets are NOT met. See "Known gaps".

Caveat on the per-agent number: the budget run uses `--graze 100` and 20k founders. That world is over its grazing capacity (about 6,554 agents) and collapses to 1,684 agents by the end of the run. At that population the per-tick cost is mostly per-cell work (fields), so us_per_agent_step is inflated. The sustained-population runs below (`--graze 400`) give 0.548 and 0.489 us, which still fail.

## Raw output

### Step 1 reference shapes

```
$ abi-bench throughput --size 256 --pop 20000 --ticks 300 --warmup 100
ticks_per_s 360.9
agent_steps_per_s 1232484
us_per_agent_step 0.811
pop_end 1684

$ abi-bench throughput --size 256 --pop 100000 --ticks 100 --warmup 50 --graze 400
ticks_per_s 33.6
agent_steps_per_s 2043244
us_per_agent_step 0.489
pop_end 16465
(wall time 5.5 s, 343% CPU; well under the 15 minute limit)

$ abi-bench speedup --size 256 --pop 20000 --ticks 200
ticks_per_s_1 78.1
ticks_per_s_8 223.0
speedup 2.85
deterministic true

$ abi-bench memory --size 256 --pop 20000
bytes_per_agent 1318
checkpoint_bytes 14131554
materials 24
```

### Evolved population

```
$ abi-bench memory --size 128 --pop 2000 --warmup 3000
bytes_per_agent 1404
checkpoint_bytes 2787903
materials 23225
```

The tick-0 figure (1318) covers founders only (hidden 4..13). The evolved figure is the mean over the population alive after 3000 ticks; a single hidden-24 genome alone is about 2.35 KB, so populations that evolve toward the maximum hidden size will exceed the 2048 budget.

### Default shape (pop 9800, graze 100)

```
$ abi-bench throughput --ticks 300 --warmup 100
ticks_per_s 419.1
agent_steps_per_s 1129389
us_per_agent_step 0.885
pop_end 2127

$ abi-bench speedup --ticks 200
ticks_per_s_1 119.6
ticks_per_s_8 309.0
speedup 2.58
deterministic true

$ abi-bench memory
bytes_per_agent 1318
checkpoint_bytes 7238013
materials 24
```

### Sustained population (20k founders, graze 400)

```
$ abi-bench throughput --pop 20000 --graze 400 --ticks 300 --warmup 100
ticks_per_s 164.2
agent_steps_per_s 1825446
us_per_agent_step 0.548
pop_end 15464

$ abi-bench speedup --pop 20000 --graze 400 --ticks 200
ticks_per_s_1 60.5
ticks_per_s_8 159.6
speedup 2.64
deterministic true
```

## Per-phase breakdown (`abi-bench phases`, via `World::step_timed`)

Wall-clock per tick, averaged over the measured ticks. `boundary_deferred` and `boundary_mint` are the sequential boundary pass (steps 4 and 5 of the tick). `fields_events` is diffusion, temperature relaxation and external events.

Default shape, 8 threads (`phases`, 300 ticks, warmup 100, pop_end 2127):

| phase | ms/tick | share |
|---|---|---|
| sort_agents | 0.308 | 12.9% |
| decide | 0.370 | 15.4% |
| chunk_tasks | 0.327 | 13.6% |
| boundary_deferred | 0.170 | 7.1% |
| boundary_mint | 0.176 | 7.3% |
| children_append | 0.029 | 1.2% |
| fields_events | 1.020 | 42.5% |
| total | 2.400 | |

Default shape, 1 thread (`phases --threads 1`):

| phase | ms/tick | share |
|---|---|---|
| sort_agents | 0.270 | 4.1% |
| decide | 1.608 | 24.4% |
| chunk_tasks | 1.346 | 20.4% |
| boundary_deferred | 0.114 | 1.7% |
| boundary_mint | 0.136 | 2.1% |
| children_append | 0.023 | 0.3% |
| fields_events | 3.104 | 47.0% |
| total | 6.600 | |

Per-phase 1-to-8-thread speedup at the default shape: decide 4.3x, chunk_tasks 4.1x, fields_events 3.0x, sort_agents 0.9x (sequential), boundary_deferred 0.7x and boundary_mint 0.8x (sequential, and slightly slower with 8 threads running).

Brief 20k shape, 8 threads (`phases --pop 20000 --ticks 300`, pop_end 1684):

| phase | ms/tick | share |
|---|---|---|
| sort_agents | 0.423 | 14.9% |
| decide | 0.500 | 17.6% |
| chunk_tasks | 0.421 | 14.8% |
| boundary_deferred | 0.193 | 6.8% |
| boundary_mint | 0.235 | 8.3% |
| children_append | 0.035 | 1.2% |
| fields_events | 1.034 | 36.4% |
| total | 2.840 | |

20k founders, graze 400, 8 threads (pop_end 15464):

| phase | ms/tick | share |
|---|---|---|
| sort_agents | 1.956 | 31.6% |
| decide | 1.609 | 26.0% |
| chunk_tasks | 0.616 | 9.9% |
| boundary_deferred | 0.459 | 7.4% |
| boundary_mint | 0.375 | 6.1% |
| children_append | 0.112 | 1.8% |
| fields_events | 1.062 | 17.2% |
| total | 6.189 | |

100k founders, graze 400, 8 threads (`phases --pop 100000 --graze 400 --ticks 50 --warmup 50`, pop_end 54824):

| phase | ms/tick | share |
|---|---|---|
| sort_agents | 17.428 | 42.2% |
| decide | 12.943 | 31.4% |
| chunk_tasks | 1.727 | 4.2% |
| boundary_deferred | 4.128 | 10.0% |
| boundary_mint | 2.885 | 7.0% |
| children_append | 1.032 | 2.5% |
| fields_events | 1.117 | 2.7% |
| total | 41.261 | |

(The `phases` run on 100k reports a different `pop_end` than `throughput` because it covers a different tick window: ticks 50-100 against 50-150.)

## Decision

The brief's remedies are keyed to one dominant phase each:

- `forward` or `observe` dominant (remedies 1 and 2): both live inside decide, which is 15-31% of the tick across shapes. It is never the dominant phase.
- Sequential boundary pass above 25% of the tick (speedup remedy 3): `boundary_deferred + boundary_mint` is 14.4% (default), 15.1% (brief 20k), 13.5% (20k graze 400) and 17.0% (100k). Below the threshold. Under the broader reading "all sequential work" (sort + boundary + mint + births) the share is 28.5% at the default shape and 61.7% at 100k, but remedy 3 only moves minting (6-8% of the tick), so it still could not have closed the gap.

The dominant phases are `fields_events` at low population (36-43%) and the sequential `sort_agents` at sustained population (32% at 15k agents, 42% at 55k). No remedy on the brief's list targets either, so no optimization was applied. This follows the task rule: no speculative changes, and no remedies outside the list.

## Known gaps

All three gaps are open at M1. Measured at `7eb3732`.

1. **us_per_agent_step: 0.811 measured vs 0.4 budget** (0.885 at the default shape; 0.548 at 20k/graze 400; 0.489 at 100k/graze 400). At the budget run's collapsed population, `fields_events` (about 1.0 ms/tick, per-cell, independent of population) is 36-43% of the tick. At sustained population `sort_agents` (sequential) is the largest phase.
2. **speedup: 2.85 measured vs 6.0 budget** (2.58 default; 2.64 at 20k/graze 400). Phases that run sequentially (`sort_agents`, both boundary phases, `children_append`) do not speed up, `fields_events` reaches only 3.0x, and decide and chunk_tasks reach about 4.2x. On 4 performance + 6 efficiency cores, 8 threads is not 8 equal cores, so 6.0x may also be out of reach on this machine whatever the code does. Determinism holds (`deterministic true`).
3. **ticks_per_s at 100k agents: 33.6 measured vs 60 budget.** The per-step budget also fails here, so the brief's "memory-bound, record and move on" rule does not apply cleanly. In the phase run `sort_agents` is 42% of the tick (17.4 ms) and decide is 31% (12.9 ms). Reaching 60 ticks/s needs a tick under 16.7 ms.

Suggested follow-ups (hypotheses from reading the code, not measured, none applied): make `sort_agents` incremental or parallel (it rebuilds the whole `Vec<Agent>` through `Option` moves every tick and runs a global `sort_unstable`); look at `Cell` layout, since `fields_events` does `inv` linear searches through a heap `Vec` per cell per pass; revisit the brief's `forward`/`observe` remedies once decide is the dominant phase.

Passing metrics: bytes_per_agent 1318 (budget 2048) and checkpoint_bytes 14.1 MB at 20k agents (budget 100 MB; measured on the freshly built world, not after a run).
