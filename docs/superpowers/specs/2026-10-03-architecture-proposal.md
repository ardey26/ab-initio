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

## Spike results (2026-10-03)

Code: `spike/` (throwaway; Rust, rayon). Tests: mass conservation exact (integer mass), same hash on 1 vs 8 threads.

### Benchmarks (M4, 128x128 grid, 1384-param MLP brains, 60 inputs)
- 8 threads: 0.20 us/agent-step, 4.9M agent-steps/s. 1 thread: 0.79 us. Only ~4x on 8 threads:
  phase 2 (apply intents, metabolism, physics) is sequential. Must be chunk-parallelized deterministically.
- Memory: 5.6 KB/agent, 98% is f32 genome. Quantize weights (i8/f16) for the real core.
- Stable populations: 10k-16k agents on 16k cells at 8-14 ms/tick (2 threads, 4 runs contending).
- Artifact intern table grew to 2k-28k entries in a few thousand ticks. GC of unused artifacts is mandatory.

### Step 1: chemistry richness (100 seeds, 24 base materials)
- 57% of base pairs beat both parents on nutrition, hardness or energy by >0.05 (74% when hot).
- 33 of 60 seeds have a plant+X combination with >+0.15 nutrition over raw plant; several reach 1.0 (clamp ceiling).
- Adjacent possible grows with depth (343 -> 1329 -> 1597 -> 1947 new useful artifacts per layer) but
  the [0,1] clamp is an artificial ceiling that closes it. Needs an unbounded or soft property scale.
- Verdict: chemistry richness is NOT the binding constraint.

### Step 2: selection (4 seeds with real payoff, 60k ticks, 10k-16k agents)
Seed 43: plant+material 14 at ambient temperature = nutrition 1.0 vs raw plant 0.417, one step, no heating.
Seed 14: one heat then combine -> 1.0. Seed 2: one heat -> 0.77. Seed 0: three heats -> 1.0.
Result in every seed: combine rate falls from 12.5% (random) to <0.5%; heat, strike, give, move all
fall below 1%; take rises to 83-89%. Artifact energy fraction -> 0. Populations converge on "sit and graze".
Random-policy baseline goes extinct within 5k ticks, so the evolved populations are under real selection.

Diagnosis: a blind combine has negative expected value (it destroys edible plant and usually yields worse
food). The payoff needs the right partner (3% of cells, specific property target), so partial progress is
never rewarded. Pure evolution of cost-bearing actions extinguishes exploration. This is the JaxLife failure.

### Ecology findings along the way
- Autotroph viability depends on seed (toxicity vs nutrition). Dead worlds are common unless the
  generator guarantees a net-positive autotroph. Decide whether dead worlds are a feature.
- Closed matter without circulation starves life: soil migrated to infertile cells and locked up.
  Soil diffusion fixed it. The real core needs explicit matter transport (water, erosion, decomposition).
- Reproduction threshold vs food energy density controls boom-bust. Needed two rounds of tuning.

### Riskiest assumption, revised
Not chemistry richness. The discovery gradient: can agents reach the first useful combination at all.
Two physics-consistent remedies to test next, in this order:
1. Environmental chemistry: reactions happen in cells (cold weathering slowly everywhere, hot near
   volcanic cells or fire) without agents. Artifacts then exist before anyone makes them. Rung 1 is
   "eat found artifacts" (one bias weight). Rung 2 is "carry materials together". Rung 3 is "heat".
   Each rung is rewarded on its own. Consistent with "the universe does not depend on life".
2. Lifetime reward-modulated learning (energy delta as reward) so a lucky individual discovery is
   repeated within a life, then imitation so it spreads. The Baldwin route the design already names.
