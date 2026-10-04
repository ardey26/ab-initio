# M2: observer and chronicle, design specification

Status: approved 2026-10-04. Extends `2026-10-03-ab-initio-design.md` section 7. Where this document
and the parent spec differ, this document wins for M2.

## 1. Purpose

A separate program that reads a finished or running simulation and writes a chronicle: factual,
dated entries about the world, its lineages and individual lives, named where possible in the
agents' own signals. It also ranks what deserves attention right now. It never writes to the core.

M2 has no viewer. The chronicle is read through a CLI and as files. M3 renders it.

## 2. Decisions that differ from the parent spec

1. **Input is checkpoints, not a snapshot format.** The observer loads the core's zstd checkpoints
   through `abi_core::checkpoint::load`. `abi-snapshot` and the live ring buffer move to M3.
   `abi-observer` therefore depends on `abi-core` (read-only use). Section 4 of the parent spec is
   amended accordingly.
2. **Behaviour is observed by tracing a replayed slice.** A checkpoint holds state, not actions.
   For each checkpoint the observer replays the next `slice` ticks (default 200) with tracing on and
   analyses the trace. This is deterministic and costs compute, not core changes beyond one hook.
3. **Lineage is carried by the core.** Every agent gets `founder: u64`, inherited at birth, set to
   the agent's own id for founders and seeded organisms. Lineage identity is then exact across
   checkpoints of any spacing. Splits within a founder group are detected by the observer.

## 3. Core changes the observer needs (delivered in the prelude plan or M2 task 1)

- `Agent.founder: u64` (checkpoint format version 3).
- `World::step_traced(&mut self, trace: &mut Vec<TraceEvent>)`: identical history to `step`,
  additionally recording one event per applied action and per birth, death, combine, give, strike
  hit, and emitted signal. Events are collected per chunk and appended in chunk order, then the
  boundary pass appends its own, so the trace is deterministic. `step` keeps delegating to
  `step_timed`; `step_traced` wraps the same body with a sink.
- `TraceEvent { tick: u64, agent: u64, founder: u64, kind: TraceKind, cell: u32, held0: MatId,
  held0_props_bins: [u8; 2], context: u8 /* bit0 bedrock here, bit1 hot cell, bit2 other agent here,
  bit3 fertile */, value: f32 /* signal[0], damage, energy given */, value2: f32 /* signal[1] */,
  other: u64 /* target agent or product material */ }` and
  `TraceKind { Move, Take, Drop, Combine, Heat, Strike, Give, Emit, Birth, Death }`.
  Material ids never reach the chronicle text; they are rendered from properties.

## 4. Crate layout

`crates/abi-observer` (lib + bin). Modules:
- `input`: run directory discovery (`checkpoints/`, `metrics.csv`, `events.log`), checkpoint
  iteration in tick order, slice replay with tracing.
- `lineage`: founder groups, population per founder per checkpoint, birth/decline/end detection,
  split detection (genome-distance clustering within a founder group), the last-witness check.
- `settlement`: DBSCAN over agent positions on the torus, persistence across checkpoints by
  centroid matching.
- `convention`: material x action x context contingency tables per lineage from traces; lift
  threshold; first agent-made material per lineage.
- `signal`: mutual information between quantized emitted signals and context or next action, per
  lineage; naming from signals.
- `chronicle`: entry type, append-only JSONL store, era summaries (compaction), queries.
- `attention`: magnitude ranking of current events; recurrence matching against extinct lineages.
- `state`: observer state persisted between runs (last processed checkpoint, lineage registry,
  extinct lineage signatures, settlement registry), serialized to `chronicle/state.bin`.
- `cli`: `run`, `chronicle`, `attention`.

Dependencies: `abi-core`, `serde`, `serde_json`, `clap`, `rayon`. No others.

## 5. Detectors

Each detector consumes one checkpoint plus the trace of its slice, and the previous observer state.
Each emits chronicle entries and updates state. Thresholds are constants in one module
(`thresholds.rs`), listed with their values below.

### 5.1 Lineages
- Group living agents by `founder`. Record per checkpoint: population, mean genome hash distance
  (Hamming over i8 weights normalised by length), territory (bounding box on the torus), top three
  eaten materials from the trace.
- **Birth**: a founder id first seen with population >= `LINEAGE_MIN = 20` becomes a lineage entry
  (scale: lineage). Founders below the minimum are "small groups" and only counted.
- **Decline**: population falls below 25% of its running maximum.
- **End**: population reaches 0. On end, scan every living agent's social memory for any member id
  of the ended lineage. If none is found, append the fact "no living agent holds a memory of any
  member of this lineage" (scale: world). This is the last-witness entry.
- **Split**: within one founder group, run k-means with k = 2 on genome weight vectors (dequantized)
  across three consecutive checkpoints; if both clusters exceed `LINEAGE_MIN` and the between-cluster
  distance exceeds `SPLIT_DISTANCE = 0.35` (fraction of weights differing by more than one
  quantization step) for all three checkpoints, record a split and start tracking the smaller
  cluster as a sub-lineage identified by (founder, split index). Sub-lineages are tracked by genome
  nearest-centroid assignment thereafter.

### 5.2 Settlements
- DBSCAN over living agent positions with torus distance, `EPS = 3` cells, `MIN_PTS = 12`.
- Clusters are matched to the previous checkpoint's clusters by nearest centroid within `EPS * 2`.
  A cluster persisting for `SETTLEMENT_PERSIST = 3` checkpoints becomes a settlement entry (scale:
  world) with its dominant lineage and the cell's dominant loose material rendered by properties.
  Disappearance after persistence is an entry.

### 5.3 Material conventions
- From the trace, per lineage: counts of (held material bin pair, action, context) for Strike,
  Combine, Heat, Give, Take. The material key is the quantized property bin key, not the id.
- Lift = P(material | action, context) / P(material). An entry is written when a cell of the table
  has at least `CONVENTION_MIN_COUNT = 30` events and lift >= `CONVENTION_LIFT = 4.0`, and the
  same cell exceeded the threshold in the previous checkpoint's slice too. Entry text names the
  material by properties (or by signal name if one exists), the action, and the context.
- **First agent-made material**: the first Combine event per lineage whose product bin was never
  seen in any earlier trace or checkpoint inventory is an entry (scale: lineage). The product's
  material id is remembered so later lineages reusing it are not credited again.

### 5.4 Signal conventions and naming
- Emitted signals are two floats in [-1, 1]; quantize each to 4 levels, giving 16 symbols.
- Per lineage, over the slice: mutual information between the symbol and each of: context bits, the
  emitter's next action, the material bin held. MI is computed from counts with a plug-in estimator
  and a bias correction of (k-1)(l-1)/(2N ln 2).
- A convention is recorded when MI >= `SIGNAL_MI = 0.5` bits with at least `SIGNAL_MIN_COUNT = 50`
  emissions in that lineage and the same symbol-to-target pair leads in two consecutive slices.
- **Naming**: a symbol maps deterministically to a syllable: consonant from the first level
  (k, t, m, r), vowel from the second (a, i, o, u). A convention binding a symbol to a material bin
  gives that material a name in that lineage's entries, e.g. "ta". Two-symbol sequences within
  5 ticks by the same agent form two-syllable names. Names are per lineage; the chronicle writes
  "which the <lineage> call 'tari'".

### 5.5 Novelty
- Read from `metrics.csv`: distinct materials and distinct behaviours per window. The observer
  writes one world-scale entry when a 10-window moving average changes by more than 25%, and a
  note in the era summary. No new computation.

### 5.6 Attention
- Every checkpoint, compute a magnitude for each candidate: lineage population change (relative),
  lineage birth or end, first agent-made material, settlement appearance or loss, bedrock
  exhaustion in a region (sum of bedrock over a 32x32 chunk reaching 0 for the first time), external
  event (from `events.log`), novelty step.
- Magnitude is a unitless score: relative change times log population for lineage events; 1.0 for
  firsts and ends; event radius squared over grid area for external events. The top `ATTENTION_N = 5`
  are written to `chronicle/attention.json` with tick, subject, score and a one-line fact. No
  valence words appear anywhere.

### 5.7 Recurrence
- When a lineage ends, store its signature: normalised vector of material-bin usage by action, and
  its signal-to-target MI table. When a new lineage is born, compare its signature after
  `RECURRENCE_AFTER = 5` checkpoints to all extinct signatures by cosine similarity; if the best is
  >= `RECURRENCE_SIM = 0.8`, write a world-scale entry linking the two with the similarity value.

## 6. Chronicle

- Entry: `{ id: u64, tick: u64, scale: World|Lineage|Life, subjects: Vec<Subject>, kind: &str,
  fact: String, evidence: Map<String, f64>, name: Option<String> }`. `Subject` is one of
  `Lineage(founder, split)`, `Agent(id)`, `Settlement(id)`, `Region(chunk)`, `Material(bin key)`.
- Facts are templated sentences in the past tense with no evaluative words. A denylist in
  `chronicle/style.rs` (good, bad, great, tragic, sadly, finally, unfortunately, impressive) is
  enforced by a unit test over every template.
- Store: `chronicle/entries.jsonl` append-only. `chronicle/eras/<start>-<end>.json` summaries are
  produced by compaction when entries older than `ERA_TICKS = 1_000_000` exceed `ERA_MAX_ENTRIES =
  10_000`: per lineage, counts of each entry kind, first and last ticks, names given, materials in
  use; the detailed entries of that era are then moved to `chronicle/archive/` (never deleted;
  disk growth is linear in events, bounded by the retention policy in the parent spec's spirit:
  a configurable `archive_budget_bytes`, oldest archives deleted first, summaries kept forever).
- Query: by tick range, scale, subject, kind.

## 7. CLI

- `abi-observer run --run <dir> [--slice 200] [--threads N]`: processes checkpoints not yet in
  the state, in order; idempotent; safe to run while `abi-sim` is still writing (it only reads
  completed files and ignores `.tmp`).
- `abi-observer chronicle --run <dir> [--scale ..] [--lineage <founder>] [--since <tick>] [--kind ..]
  [--json]`: prints entries, newest first by default.
- `abi-observer attention --run <dir>`: prints the current attention list.

## 8. Performance

Observer budget per checkpoint at 256x256 and 20k agents: slice replay of 200 ticks (about 4 s at
M1 speed), detectors under 2 s. Observer runtime must stay under the sim's checkpoint interval so
it can keep up with a live run. DBSCAN uses a grid index; k-means runs on at most 2,000 sampled
agents per founder group. Observer state under 50 MB.

## 9. Testing

- Unit: MI estimator on known distributions; DBSCAN on synthetic clusters on a torus; lift table;
  naming determinism; the no-valence denylist over all templates.
- Integration: generate a 128x128, seed 6, 12k-tick run with checkpoints every 2000 ticks in a
  temp dir (about 2 minutes), run the observer, assert: at least one lineage entry, lineage
  populations sum to the checkpoint population, running the observer twice produces no new
  entries, `chronicle --json` round-trips.
- Determinism: the observer run over the same run directory yields byte-identical `entries.jsonl`.

## 10. Milestone exit

M2 is done when: `abi-observer run` on a 60k-tick seed 6 run produces lineage, convention and
attention entries without manual input; the first agent-made material entry appears on a seed that
transitions; the chronicle contains no evaluative words; and the observer keeps pace with a live
run at 256x256.
