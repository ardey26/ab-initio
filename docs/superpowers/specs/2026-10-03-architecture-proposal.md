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

## Spike 2: environmental chemistry (2026-10-03)

Changes: cells react on their own (hot cells every tick, cold cells once per 32 ticks on average),
2% volcanic cells at temperature 0.8, bedrock erodes into a seeded ore at 10 mass/tick, material
identity = quantized property vector (15 levels) so recipes landing on the same properties are one material.

Without quantized identity the artifact table hit 8.8M ids in 3000 ticks. With it: 60k-80k ids at 40k ticks,
still growing slowly. GC remains mandatory.

Results, 40k ticks (~60 generations, generation time ~600-700 ticks), 10k-15k agents:
- Seed 0: energy from artifacts 9.1% -> 1.3% and plateaus. Combine 7.6% -> 1.6%. Heat 4.0% -> 0.35%.
  Movement stays at 32% (was 10% without environmental chemistry).
- Seeds 43, 14: artifact energy -> 0.1%, combine -> 0.5-0.7%, heat -> 0.3-0.4%.
- Seed 2: extinct at 4k ticks (erosion + weathering shifted the ecology).
- Environmental reactions fall 4.7M -> 0.8-1.5M per window: agents graze plant below the reaction
  threshold and eat the ores. Life starves the chemistry of inputs.

Verdict: a small persistent artifact niche appears (seed 0) but nothing climbs. Found artifacts with
nutrition 1.0 are not enough for selection to hold exploratory actions. Direction of selection at the
start is against combining in every seed. 60 generations is too few to rule out drift finding it later;
a 1M+ tick run is needed as the control for any further redesign.

Recommended next: (1) NK-landscape chemistry with tunable ruggedness K and unbounded log-scale
properties, so "for which K does culture emerge" is the experiment; (2) lifetime reward-modulated
learning + imitation; (3) a multi-hour control run of the current design.

## Spike 3: lifetime learning + NK-landscape chemistry (2026-10-03)

Changes: properties are unbounded reals, squashed where physics/brains read them (no ceiling). Each
output property = random Fourier function (GP sample) of K other properties from both reactants plus
continuous temperature. Lifetime copy of the network; action head learns by reward-modulated eligibility
traces (reward = energy delta minus the agent's running baseline); learning rate and imitation gain are genes.

Richness per K (60 seeds): nearly flat (improving pair fraction 0.39-0.42); best nutrition climbs to
depth 4 at every K (0.79 at K=0, 0.93 at K=7). Headroom now limited only by the squash.

Seed 7, 40k ticks, 14k-16k agents:
| run              | combine 4k->36k | artifact energy 4k->36k | take at 36k |
| K=0 learn        | 5.8% -> 0.4%    | 5.1% -> 0.15%           | 72% |
| K=2 learn        | 6.6% -> 0.5%    | 7.7% -> 0.26%           | 75% |
| K=7 learn        | 9.1% -> 0.6%    | 11.8% -> 0.3%           | 70% |
| K=2 no learning  | 8.6% -> 1.4%    | 8.8% -> 0.6%            | 83% |
Learning-rate gene never selected upward (~0.0007). Lifetime learning did not help; K did not matter.

Material explosion again: continuous temperature + unbounded properties gave 1-3M materials in use per
window and 50-130 ms/tick from memory pressure. Fixed for spike 4 by clamping raw properties to [-6,6],
4 temperature buckets, and 5 quantization levels per property (material lattice bounded at 390k).

Diagnosis after three negatives: every configuration converges on "sit on a fertile cell and take plant
as it regrows", which fed 15k agents every time. Alternatives are selected under scarcity of the default
resource, not under abundance of the alternative. Spike 4 tests scarcity (plant regrowth 8 and 4 vs 40).

## Spike 4: scarcity (2026-10-03) -- FIRST POSITIVE RESULT

Seed 7, K=2, no lifetime learning, 40k ticks. Plant regrowth 40 -> 8 and 4 mass/tick per fertile cell.

| run       | pop 4k -> 40k | combine 4k -> min -> 40k | artifact energy 4k -> min -> 40k | move at 40k |
| growth 8  | 3.1k -> 4.7k  | 11% -> 8% -> 13%         | 20% -> 14% -> 44%                | 42% |
| growth 4  | 1.5k -> 8.0k  | 7% -> 0.9% -> 26%        | 13% -> 4% -> 96%                 | 28% |
| growth 8 + learning | 3.0k -> 3.1k | 4.8% -> 0.3% | 9.6% -> 1.5%                 | 49% |

Growth 4 is a transition: combining decays exactly as in spikes 1-3 for ~16k ticks (~20 generations),
then a lineage that lives on what it makes sweeps the population. Energy from artifacts goes to 96%,
population grows 5x past the grazing carrying capacity, 57k distinct artifacts in use per window.
Growth 8 shows the same transition, slower and partial. Growth 40 (spikes 1-3) never transitions.

Lifetime reward learning as implemented is anti-exploratory: it suppresses combining (0.3%) and
evolution selects the learning-rate gene down. Immediate-reward learning locks in the grazing attractor.

Verdict: the primitive set and generated chemistry are sufficient. The binding constraint was ecology:
alternatives are selected under scarcity of the default resource. Replication on seeds 2, 6, 0 pending.

Replication (growth 4, K=2, no learning, 40k ticks):
- Seed 2: transition by 8k ticks. Artifact energy 53% -> 96%, combine 12% -> 22%, pop 2.8k -> 4.6k. Agent combines 2.1M/window vs environmental 2.2M.
- Seed 6: slow partial transition. Artifact energy 18% -> 55% and rising at 40k, combine 5.7% -> 13%, pop 1.3k -> 2.3k.
- Seed 0: extinct at 4k. Plant nutrition 0.30 at growth 4 cannot sustain the founding population. Scarcity must be
  tuned per world or the generator must guarantee a viable autotroph; dead worlds otherwise dominate.
Result replicates: 3 of 3 surviving seeds transition to artifact-based life under scarcity; 0 of 7 runs did under abundance.
