# ab-initio: architecture proposal and first spike

Status: proposal, approved for spike only. Not yet a spec.

## Machine (2026-10-03)

- Apple M4, 10 cores (4P + 6E), 24 GB unified memory, 10-core GPU (Metal 4), 70 GB free disk, macOS 26.6.
- Expected: 50k-100k agents at 50-100 ticks/s on CPU with ~1k-param brains.
- Bottlenecks: 4 fast cores, laptop thermals over multi-day runs, disk for checkpoints (budget 20 GB).
- GPU for brain inference is a later experiment. CPU first.

## Architecture

Three processes, one snapshot format.

1. Core (Rust, headless). Fixed timestep. Two-phase tick: (1) all agents read a frozen
   world and emit intents, in parallel; (2) intents applied in deterministic order (cell, agent id).
   Per-agent counter-based RNG keyed on (seed, tick, agent id). Thread count never affects history.
   Checkpoints every N ticks, deltas to a ring buffer.
2. Observer (separate process). Reads snapshots, writes chronicle. Never writes to core.
3. Viewer (separate process, wgpu/Metal). Attaches to ring buffer or any checkpoint. Past = nearest checkpoint + replay.

World: 2D heightfield grid. Cells hold a material inventory, temperature, water, elevation.
Geology/climate/impacts are field updates independent of agents. Mass is integer (exact conservation).

Chemistry: per seed, 32-64 base materials, each a property vector (density, hardness, melt point,
energy, brittleness, solubility, nutrition, toxicity). Combination = seed-generated function of two
property vectors + conditions (temperature, impact) -> result vector. Artifacts are interned
composition trees; unbounded nesting, bounded storage via interning + GC. Mass = sum of parts.

Agents: genome = body properties, brain dims, initial weights, local learning rates, imitation gain.
Brain: MLP/GRU < 2k params. Inputs are property summaries, never material ids.
Lifetime learning: local Hebbian rule with evolved rates. Imitation: plasticity nudge toward an
observed neighbour's action. Social memory: fixed 8-16 slot table per agent.

Fidelity: idle regions tick every k steps with k-scaled effects; mass and energy sums exact at every fidelity.

## Primitive actions

move, take, drop, combine, heat, strike, give, emit.

Not actions (physics instead):
- Eating: held material with nutrition above the agent's evolved digestion threshold is metabolized automatically.
- Reproduction: asexual division above an energy threshold; sexual = give of gamete material.

Derived, unnamed in code: strike(hard item, hard cell) -> fragments (mining); strike(artifact) -> parts;
heat(flammable cell) above ignition -> persistent fire.

## Riskiest assumption

Not "is the chemistry rich" alone. It is the gradient: can near-random evolved policies climb from
no combining to a first useful combination when payoff needs a multi-step sequence.

Spike, in order:
1. Chemistry alone, 100 seeds: fraction of pairs that beat both parents on a fitness-relevant
   property; depth of improving chains (adjacent possible growth vs. regression to the mean).
2. Chemistry + selection: 5k-10k agents, random-init brains, ~500 generations, headless.
   Does combine usage rise above a random-policy baseline and persist?
   Pass 1 + fail 2 => credit assignment problem, redesign the gradient, not the materials.

The spike is also the first benchmark harness (agent-steps/s, memory/agent) and the first
novelty metric (distinct artifacts + behaviours in use per window).

Decomposition after the spike: core + benchmarks, observer, viewer, time travel.
