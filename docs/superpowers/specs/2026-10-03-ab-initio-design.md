# ab initio: design specification

Status: draft for review. Supersedes the proposal in `2026-10-03-architecture-proposal.md`, which
remains as the record of the spike series.

## 1. Purpose

An open-ended artificial life simulation in which intelligence, culture and technology emerge from
physical primitives and are never authored. The player is the Watcher: all-seeing, rarely acting,
outside the causal loop. There is no goal, win state or fail state.

Three programs share one history: a performance-critical headless core, a lightweight viewer that
attaches and detaches without slowing the core, and an observer that reads the history and writes a
chronicle. Runs are deterministic per seed, run unattended for days on one consumer machine, and can
be revisited at any past moment.

### Non-goals for v1
- Sexual reproduction, 3D terrain, multi-machine distribution, any language model.
- GPU simulation: deferred until CPU budgets in section 9 are met and profiling shows brain inference
  dominates. Unified memory on this machine keeps the option cheap.
- Authored content of any kind: no item, species, technology, faction or event types.

## 2. Principles (binding)

1. Physics, not nouns. The code contains materials with properties, agents with genomes, cells with
   fields, and eight actions. Tool, tribe, war, ritual and leader are patterns the observer may detect.
2. Closed matter, open energy. Mass is integer and exactly conserved at every tick and every fidelity.
   Energy enters from the star and leaves as heat.
3. The universe does not depend on life. Geology, climate, chemistry and impacts run with zero agents.
4. Extinction is not an end state. Abiogenesis can restart life when conditions allow.
5. Deterministic per seed, independent of thread count.
6. Bounded memory over multi-day runs.
7. Performance is a design input. Every feature is weighed against agent-steps per second.

## 3. Requirements learned from the spikes

These are hard requirements. Each one, when absent, killed emergence in a measured run.

R1. Scarcity of the default resource. Autotroph regrowth must be calibrated per world so that grazing
    alone supports far fewer agents than the world's chemistry can. Under abundance (regrowth 40) zero of
    seven runs developed technology; under scarcity (regrowth 4) three of three surviving runs did.
R2. Matter transport. Soil (and water) must diffuse between cells. Without it matter locked up in
    infertile cells and plant stock fell 95% and never recovered.
R3. Bounded material identity. A material is its quantized property vector. Without this the artifact
    table reached 8.8M entries in 3000 ticks.
R4. Lifetime learning must not be plain immediate-reward learning. That variant suppressed exploration
    and evolution selected its gene to zero. Learning and imitation ship behind genes that can be zero
    and behind a configuration flag, as experiments, not as load-bearing mechanisms.
R5. Viable autotroph per world. The generator guarantees the star-fed material is net-positive food.
    Dead-on-arrival worlds are not interesting to watch. Death later is.
R6. Take is a property-space query. The brain emits a target property vector and takes the nearest
    item. Argmax-by-one-property made partner selection impossible.

## 4. System decomposition

Cargo workspace, Rust stable, no unsafe outside the snapshot mmap layer.

| crate | kind | responsibility |
|---|---|---|
| `abi-core` | lib | world, chemistry, agents, physics, tick, checkpoint, replay, external events. No I/O except through traits. |
| `abi-sim` | bin | headless runner: loads config, runs core, writes checkpoints, publishes the live ring buffer, writes metrics. |
| `abi-snapshot` | lib | versioned flat binary snapshot and delta formats; mmap readers; shared by all three programs. |
| `abi-observer` | lib + bin | pattern detectors, chronicle store, naming, attention ranking. Reads snapshots only. |
| `abi-view` | bin | wgpu diorama viewer and UI. Reads snapshots and ring buffer; sends interventions as external events to `abi-sim`. |
| `abi-bench` | bin | benchmark and regression harness, including the emergence regression test. |

Dependency direction: view and observer depend on snapshot only. sim depends on core and snapshot.
Nothing depends on view or observer.

Processes communicate through files and a shared-memory ring buffer in a run directory:
`runs/<seed>-<name>/{config.toml, events.log, checkpoints/, ring.shm, metrics.csv, chronicle/}`.

## 5. Core model (`abi-core`)

### 5.1 World
- 2D torus grid of cells, chunked 32x32. Default 256x256, target 512x512.
- Per cell: elevation (f32, from a seeded heightfield, changed by uplift and erosion), water (u32 mass,
  conserved, part of matter), temperature (f32), ambient temperature (f32, volcanic cells high),
  bedrock mass (u32) and ore material id, loose inventory (small vector of (material id, u32 mass)),
  fertility (derived each tick from water and soil presence, not stored as a flag).
- Soil, water and heat diffuse to the 4 neighbours each tick with exact integer transfers.
- Mass accounting: sum over cells (loose + bedrock + water) + agents (body + held) is an invariant
  checked at every checkpoint and in debug builds every tick.

### 5.2 Star and climate
- Luminosity L(t) follows a seeded deep-time schedule: slow rise, long plateau, decline to zero.
  Autotroph growth is proportional to L(t) times local water. Finitude is a consequence, not a script.
- Climate: a drifting rainfall field (sum of seeded low-frequency harmonics in space and time) adds
  water; evaporation removes it toward a global reservoir that rains back. Ice ages are long-period
  modulations of the global temperature offset, which lowers growth and freezes water in cold cells.
- Regions diverge because rainfall, volcanism and ore are spatially correlated per seed.

### 5.3 Geology
- Uplift: seeded slow elevation increase along a few seeded ridge lines. Erosion: elevation flows
  downhill with water; bedrock releases its ore into loose inventory at a rate scaled by water.
- Volcanism: 1-3% of cells have high ambient temperature; the set drifts over geological time.
- Impacts: a Poisson process on the deterministic hash. An impact heats a disc, converts loose matter
  in it to ore and soil, and raises elevation at the rim. Mass is preserved exactly.

### 5.4 Chemistry
- NP = 8 properties: density, hardness, melting point, energy content, brittleness, solubility,
  nutrition, toxicity. Raw values are unbounded reals clamped to [-6, 6]; physics and brains read the
  squashed value in (0, 1).
- Base materials: 24 to 64 per seed, sampled with a seeded low-rank correlation structure. Three roles
  are guaranteed: a soil (inert sink), a star-fed autotroph with net-positive food value (R5), and a
  crust material with high hardness. Everything else about them is seed-random.
- Reaction: for two materials and a temperature, each output property is 0.5(a_j + b_j) plus a seeded
  random Fourier function (a Gaussian-process sample) of K other properties from both reactants and the
  temperature. K (0..7), amplitude and lengthscale are world parameters. Reactions are commutative by
  canonical ordering and deterministic.
- Identity (R3): a material is its squashed property vector quantized to Q = 5 levels per property.
  Different recipes landing in the same bin are the same material. The lattice holds at most 5^8 =
  390,625 materials. An id always denotes one bin and is never reassigned. At checkpoint boundaries the table
  drops the property and recipe data of ids with no mass anywhere and no holder, keeping only the
  bin-to-id map (8 bytes per bin). If that bin is produced again it receives its old id and its data
  is recomputed. Memory for full entries is bounded by materials that currently exist.
- Environmental chemistry: every tick, in each cell, the two most massive loose non-soil items react if
  the cell is hot, or with probability 1/32 (deterministic hash) if cold. Reaction consumes equal mass
  from both and produces the result in the same cell.
- Fire: a cell with a loose material whose energy content exceeds a threshold and whose temperature
  exceeds the material's melting point burns: mass converts to soil at a fixed rate, temperature rises,
  light is emitted (viewer glow). Fire spreads only through temperature diffusion.

### 5.5 Agents
- Storage: structure-of-arrays per chunk; agents sorted by (cell, id) each tick. Free-list reuse.
- State: position, energy (f32), age, body mass (u32), two held slots (material id, u32 mass), last
  action, 2-float signal, genome, social memory, optional lifetime weights and learner.
- Genome: body size (sets body target mass, movement cost, strike force), digestion threshold, brain
  hidden size (4..32), initial weights (i8 with per-layer scale), learning rate gene, imitation gain
  gene, exploration temperature gene. Mutation: per-weight Gaussian with probability 0.08, hidden-size
  change with probability 0.02 (weights grown with zeros or truncated), scalar genes with probability 0.2.
- Cognition cost: per-tick energy = base + c * (number of weights). Capacity only persists if it pays.
- Sensors (inputs, all property-based, never ids): own energy, age, held slot properties and masses,
  for here and 4 neighbours the max nutrition, max hardness, loose mass and temperature, properties of
  the two most massive items here, bedrock hardness here, count of agents here, the observed neighbour's
  signal, last action and the memory vector this agent holds about it.
- Actions (8): move(dir), take(target property vector), drop(slot), combine, heat, strike, give, emit.
  Costs are per action. Semantics as implemented in the spike, with: combine uses the cell temperature;
  strike resolves in this order: if the cell has bedrock and the held item is harder than the crust,
  mine ore; else if another agent is here, it loses energy proportional to the striker's held hardness
  and body size (the only way to harm another agent); else if the held item is an artifact, split it by
  recipe; else nothing; give transfers slot 0 to the lowest-id other agent here with a free slot.
- Not actions: eating (automatic digestion of held material above the digestion threshold),
  reproduction (asexual division at an energy and body-mass threshold), death (energy <= 0 or age >
  lifespan; body and held mass return to the cell).
- Social memory: 8 slots of (other agent id, 4-float vector). Updated on give, receive, strike and
  co-location: the vector accumulates energy deltas attributable to that other and recency. Replacement
  is least-recent. Social structure is whatever the observer finds in these tables.
- Lifetime learning (R4, flagged): reward-modulated eligibility traces on the action head, reward =
  energy delta minus running baseline, plus an entropy bonus scaled by the exploration temperature gene.
  Imitation: nudge toward the observed neighbour's last action scaled by the imitation gene. Both
  genes can be zero; default configuration runs with the flag off, matching the positive spike result.
- Abiogenesis: in a cell with no agents within radius 4, loose material whose energy content and
  nutrition both exceed thresholds, and temperature in a window, a minimal agent (smallest body, hidden
  size 4, random weights) forms from that mass with probability p per tick (deterministic hash). p is
  small enough that it matters only after collapses.

### 5.6 Tick (deterministic, parallel)
1. Sort agents by (cell, id); build per-cell ranges. (Parallel radix sort by chunk, deterministic.)
2. Decide: every agent reads the frozen world and emits an intent. Parallel over agents. RNG per agent
   from hash(seed, tick, id).
3. Apply, interior: each chunk applies intents of agents whose effects stay inside the chunk, in
   (cell, id) order. Parallel over chunks.
4. Apply, boundary: intents that cross chunk boundaries (move across, give across) are applied
   sequentially in (cell, id) order.
5. Metabolize, learn, reproduce: parallel over chunks; children placed in the parent's chunk or
   deferred to the boundary pass if the target cell is in another chunk.
6. Physics: compute diffusion fluxes from the frozen field (parallel), apply (parallel), then
   erosion, environmental chemistry, fire, growth, star and climate updates (parallel over chunks;
   new material ids are minted in a sequential pass over a collected list, in chunk order).
7. External events for this tick (interventions) are applied after physics, from the event log.
Floating point: fixed evaluation order, no fast-math, no reductions whose order depends on thread count.
No HashMap iteration affects simulation state.

### 5.7 Fidelity scaling
Chunks with no agents and no agents in neighbouring chunks run physics every k ticks (k in {1,4,16})
with rates multiplied by k. Integer transfers are scaled exactly. Chunks with agents always run at k=1.
Mass invariant holds at every k.

### 5.8 Time, history, checkpoints
- Checkpoint every C ticks (default 10,000) as a zstd-compressed snapshot of the full state including
  the material table. No RNG state exists to save: all randomness derives from (seed, tick, id).
  Restore reproduces the state hash exactly.
- Retention: keep every checkpoint for the last 1M ticks, every 4th before that, every 16th before
  10M, under a configurable disk budget (default 20 GB). Thinning never removes the first checkpoint.
- Any tick is reachable by restoring the nearest earlier checkpoint and replaying; worst case C ticks.
- External events (interventions) are logged with their tick and replayed, so history with
  interventions is still deterministic.
- Live ring buffer: last 4096 ticks of per-agent (position, energy, held ids) and per-cell deltas in
  shared memory for the viewer. The core never waits on readers.

## 6. Snapshot format (`abi-snapshot`)
Flat, versioned, little-endian, aligned for mmap. Sections: header (version, seed, tick, dims),
material table (id, key, 8 squashed props, recipe), cells (elevation, water, temp, bedrock, ore,
top-4 loose items), agents (id, pos, energy, age, body, held, genome hash, hidden size, signal, last
action), social memory block, optional full genomes block, stats. Readers validate version and length.

## 7. Observer (`abi-observer`)
Runs on each new checkpoint and on the live ring at a lower rate. Writes to `chronicle/` as
append-only structured entries {tick, scale (world | lineage | life), subject ids, fact, evidence,
name}. Entries are facts, never evaluations.

Detectors:
- Lineages: agents clustered by genome distance and descent (ids carry parent id in the snapshot).
  Birth, split, decline and end of lineages are entries. On end: "no living agent holds a memory of
  any member" is checked against all social memories and recorded when true.
- Settlements: spatial clusters (DBSCAN on positions over a window) with persistence.
- Material conventions: for each lineage, a material x action x context contingency table; a cell
  with lift above threshold over a window becomes an entry (e.g. material M is held when striking
  bedrock by 80% of strikers in lineage L).
- Signal conventions: mutual information between emitted signal (quantized) and context or next
  action, per lineage. Above threshold: an entry, and the signal becomes the name source.
- Naming: a quantized 2-float signal maps deterministically to a pronounceable syllable string. When
  a lineage has a convention for a material or place, the chronicle uses that name; otherwise it
  uses a descriptive phrase from properties.
- Novelty metric: distinct materials in use (held or digested) plus distinct action 4-grams
  conditioned on held-property bins, per window, per lineage and global. Written to metrics.csv
  every window from day one. A flat line fails the emergence regression test.
- Attention: ranks current events by magnitude (population change, first use, extinction, impact,
  irreversibility such as bedrock exhaustion) across the world and exposes the top entries for the
  viewer without valence.
- Recurrence: when a new lineage's material-use and signal signatures are close to an extinct one's,
  an entry links them.
Compaction: entries older than a configurable age are summarized per era per lineage; summaries keep
ids so the past can be revisited from checkpoints.

## 8. Viewer (`abi-view`)
- wgpu (Metal on macOS). Low-poly diorama on a plinth: faceted terrain from the heightfield,
  translucent animated water, drifting clouds casting shadows from the rainfall field, day/night from
  the star, fire and hot cells glow after dark.
- Everything rendered from properties, nothing from ids: material color and finish from a fixed
  seeded mapping of the property vector (hue from density and hardness, saturation from energy
  content, roughness from brittleness, translucency from solubility). Placed matter drawn from cell
  inventories by mass. Creatures from genome: body size, hidden size, digestion threshold map to
  scale, head size and hue.
- Scales: whole world, region, cell, agent. Agent inspector shows genome, live weights as an image,
  activations, social memory table, inventory properties. No explanation text.
- Time: pause, 1x to geological fast-forward (attach to live sim or replay from checkpoints), scrub to
  any checkpoint, step. The viewer never blocks the sim.
- Chronicle panel: night navy and gilt, classical serif. Attention list on top. Click an entry to
  jump to its tick and subject.
- Interventions, all as external events through physics: rain or drought over a region, temperature
  change, drop matter of a chosen existing material, impact, seed a minimal organism. Each is logged
  and visible in the world as a physical change.

## 9. Performance

Measured on the M4 in the spike (128x128, 1384-weight brains, 8 threads): 0.20 to 0.52 us per
agent-step, 4x speedup on 8 threads, 5.6 to 12.7 KB per agent.

Targets for this machine, held as budgets:
| metric | target |
|---|---|
| agent-step cost, all phases, 8 threads | <= 0.4 us |
| parallel speedup on 8 threads vs 1 | >= 6x |
| sustained | 100k agents at >= 60 ticks/s on 512x512 |
| memory per agent | <= 2 KB (i8 weights, no lifetime copy when learning gene is 0) |
| material table | <= 100 MB |
| checkpoint size / disk | <= 100 MB each, 20 GB total |
| viewer impact on sim | 0 (sim never waits) |
| determinism | identical hash at 1, 4, 8 threads on every bench run |

`abi-bench` runs on every change: ticks/s, agent-steps/s, us/agent-step, bytes/agent, ticks per
generation, novelty metric slope. Profile before optimizing; every optimization lands with its
before/after numbers in the commit message.

## 10. Testing
- Unit: chemistry commutativity, identity interning, K=0 has no epistasis, quantization bounds.
- Invariants: mass conservation every tick in debug, every checkpoint in release; checkpoint
  round-trip hash equality; thread-count determinism; replay with logged interventions equals live.
- Emergence regression: reference seeds under the scarcity configuration must reach >= 50% energy
  from artifacts within 60k ticks at 128x128. This is the test that the primitives still work.
- Benchmark regression: any target in section 9 regressing by more than 10% fails the bench.

## 11. Milestones
M1 Core + headless sim + bench + checkpoints/replay. Reproduce the spike 4 transition at 256x256
   and meet the section 9 budgets.
M2 Observer with lineages, novelty metric, material conventions, chronicle CLI.
M3 Viewer MVP: diorama, time controls, attach/detach, agent inspector.
M4 Interventions as external events, scrub to any checkpoint, chronicle panel.
M5 Deep time: star schedule, climate drift, ice ages, impacts, abiogenesis, fidelity scaling.
M6 Experiments behind flags: lifetime learning with exploration, imitation, signal conventions study.

## 12. Risks
- Scarcity calibration per world may need an automatic search (run a short grazing-only probe at
  world generation and set regrowth from its carrying capacity).
- The 390k material lattice may be too coarse for deep technology trees; raising Q to 6 gives 1.7M.
- Chunk-parallel apply with boundary pass may not reach 6x; fallback is finer chunks and double
  buffering of boundary intents.
- Observer detectors on 100k agents must be sampled, not exhaustive, to stay lightweight.
