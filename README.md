# ab initio

An artificial-life simulation in which intelligence, culture and technology are meant to emerge
from physics rather than be written in. You are the Watcher: you can see everything, revisit any
moment, and rarely act. There is no goal, no score and no ending.

This repository holds the simulation core (milestone M1, complete) and the design for the observer
that will narrate it (M2, planned). There is no picture yet. The viewer is M3.

## What it is, in one paragraph

A small planet runs on a grid. Every cell holds materials, each material is a short list of
physical properties, and materials react with each other according to rules generated from the
world's seed. Creatures with tiny evolved brains move around and can do eight things: move, take,
drop, combine, heat, strike, give, and make a sound. They eat whatever they hold that their body
can digest, they split in two when they have enough energy and mass, and they die when they run
out. Nothing in the code knows what a tool, a tribe or a ritual is. Those words belong to the
observer, which looks at the data afterwards and reports patterns as facts.

## The ideas underneath, explained plainly

Most of what makes this project work is a handful of mathematical choices. None of them needs
more than high-school maths to understand.

### 1. Same seed, same history, on any number of CPU cores

Simulations use randomness: which way a creature turns, how a mutation lands. Ordinary random
number generators hand out numbers in sequence, so if two threads ask at the same time the order
changes and the history diverges. We never use a sequence. Every random decision is computed as a
**hash of three numbers**: the world seed, the current tick, and the identity of whoever is asking.
A hash is a fixed scrambling function: the same three inputs always give the same output, and
nearby inputs give unrelated outputs. So creature 4,021 at tick 88,000 gets the same "random"
number whether the machine has one core or ten, and whether it computed it first or last.

The tick itself is arranged so that order cannot matter. The grid is cut into 32-by-32 chunks.
Each chunk is updated by one thread that may touch only its own cells and its own creatures.
Anything that crosses a chunk boundary (a creature walking into the next chunk, a reaction that
creates a new material) is written to a list and applied afterwards, in one fixed order. The test
suite proves this: the complete world state is boiled down to a single 64-bit fingerprint and
compared across 1, 4 and 8 threads, and against a stored value, after thousands of steps.

### 2. Matter is counted in whole numbers and never lost

Every quantity of material is an integer. Floating-point numbers drift: add a tenth ten times and
you do not get exactly one. Integers do not drift. Every action moves whole units from one place
to another, so the sum over the entire world is the same number every tick, and the program checks
it. Energy is different: it enters from the star and leaves as heat, so it is allowed to float.

Spreading soil and water between neighbouring cells needs care, because in parallel two threads
might both try to change the same cell. The solution is a **two-pass flux**: first every cell
computes, from a frozen copy of the world, how much it will send to its right and lower
neighbours; then every cell applies its own outflows and the inflows its left and upper neighbours
computed. Each cell is written by exactly one thread, every unit sent is a unit received, and the
result does not depend on which thread ran first.

### 3. A material is its properties, not its name

Materials have eight properties: density, hardness, melting point, energy content, brittleness,
solubility, nutrition and toxicity. Each is a number that can be any real value, but the
simulation reads it through a squashing function that maps the whole number line onto the interval
from 0 to 1 (large negative becomes near 0, large positive near 1). Then it rounds each squashed
property to one of five levels. The eight rounded levels together are a material's **identity**.

This does two things. It bounds the universe of materials: five levels to the power of eight is
390,625 possible materials, so memory cannot grow forever however much the world experiments. And
it makes different recipes converge: two reactions that produce nearly the same stuff produce the
*same* material, which is how a technology can be rediscovered by two lineages independently.

### 4. Chemistry is a random landscape with a ruggedness dial

When two materials react, each property of the result is the average of the parents' properties
plus a wobble. The wobble is a **random smooth function**: a sum of a few cosine waves with random
frequencies, phases and heights, chosen once from the seed. Mathematicians call a function built
this way a sample from a Gaussian process; the practical meaning is that it is smooth (similar
inputs give similar outputs) but unpredictable (you cannot guess it without trying it).

Each output property is allowed to depend on K other properties of the two parents, plus the
temperature. K is a dial. At K = 0 each property evolves on its own and the landscape is gentle.
At K = 7 everything depends on everything and the landscape is rugged. This borrows directly from
Stuart Kauffman's NK model of fitness landscapes, which is where the question "how rugged can a
world be before evolution stops finding anything" was first asked.

Because properties are unbounded before squashing, there is no ceiling: a material can always be
harder or more nourishing than any that exists, so the set of things worth discovering never
closes. Kauffman called this the adjacent possible.

### 5. Brains are small, cheap and evolved

A creature's brain is a tiny neural network: 72 inputs (what it holds, what is nearby, what the
neighbour just did, what it remembers about that neighbour), a hidden layer of 4 to 24 units, and
24 outputs. The weights are stored as single bytes, which is why a whole genome fits in about
1.3 kilobytes. There is no training. Children copy their parent's weights with small random
changes, and natural selection does the rest: brains that keep their owner fed reproduce.

Choosing an action from the network's outputs uses the **Gumbel-max trick**: add a particular kind
of random noise to each output and pick the largest. This is exactly equivalent to sampling from a
softmax distribution but needs only one pass and, importantly, consumes randomness from the hashed
generator above, so it stays reproducible.

Every weight costs a little energy per tick. Capacity is only kept if it pays for itself.

### 6. What we learned from four prototype experiments

Before building the real engine we ran a throwaway prototype and tested the assumption that
seemed riskiest: that generated chemistry is rich enough to reward combining things. It was. The
real obstacle was different, and it took four experiments to find it:

| experiment | result |
|---|---|
| Rich chemistry, generous food | Evolution eliminated combining. Grazing fed everyone. |
| Plus artifacts lying around in the environment | Same decline, slower. |
| Plus lifetime learning and a rugged chemistry dial | Learning made it worse; evolution switched the learning gene off. |
| **Plus scarcity: regrowth cut tenfold** | Populations switched to living on what they combined. 96% of their energy came from artifacts, with nested recipes and no single product dominating. |

Nothing is invented while grazing feeds everyone. That finding became a hard requirement of the
design: autotroph regrowth is calibrated per world so that grazing alone can support only about
100 creatures per 1,000 cells.

### 7. How the observer will read the world (M2)

The observer never touches the simulation. It loads saved checkpoints, replays a short slice with
tracing on, and runs statistics. Three of them deserve a plain description:

- **Mutual information** measures how much knowing one thing tells you about another, in bits. If
  a lineage's creatures make sound "ti" mostly when holding a particular material, the mutual
  information between sound and material is high, and the observer records that the lineage
  "came to call" that material "ti". The name is derived from the sound, not invented by us.
- **Density clustering (DBSCAN)** finds groups of creatures that stay close together on the torus
  and keeps track of whether the group persists. That is how a settlement becomes a chronicle
  entry without anyone defining the word.
- **Lift** compares how often a material is used in a context to how often it is used in general.
  A material held when striking bedrock four times more often than chance, two slices in a row,
  becomes a recorded convention.

The chronicle only states facts in the past tense. A test fails the build if any template contains
an evaluative word.

## Repository layout

```
crates/abi-core      the simulation: chemistry, world, agents, tick, events, checkpoints
crates/abi-sim       command line runner: run, replay, resume (M2 prelude), verify
crates/abi-bench     throughput, memory, speedup, per-phase timing, emergence canary
spike/               the throwaway prototype (reference only, not built)
docs/superpowers/    specifications and implementation plans
docs/perf.md         measured performance against budgets
```

## Running it

```bash
cargo build --release

# simulate 60k ticks of seed 6 on a 128x128 world, checkpoints every 10k ticks
./target/release/abi-sim run --seed 6 --size 128 --pop 1000 --ticks 60000 --out runs/first

# revisit any past moment: loads the nearest checkpoint and replays forward, bit-exact
./target/release/abi-sim replay --run runs/first --to 42000

# prove the history does not depend on thread count
./target/release/abi-sim verify --seed 6 --ticks 500 --size 64 --pop 300

# the emergence canary (about 11 minutes): do the primitives still produce artifact-based life?
cargo test --release -p abi-bench -- --ignored
```

`runs/first/metrics.csv` has one row per 1,000 ticks: population, action rates, energy from
artifacts, distinct materials and behaviours in use (the project's health metric), and more.

## Status

M1 complete: deterministic parallel core, exact conservation, replay, checkpoints, metrics,
benchmarks, and the emergence canary passing on 4 reference seeds. Known gaps, all measured and
recorded in `docs/perf.md`: per-step cost and parallel speedup are about half the targets, and the
transition to artifact-based life happens on roughly one seed in five rather than every seed. The
M2 prelude plan addresses the first two; an experiment harness in the same plan is designed to
explain the third.

## Reading

The project's ideas come from JaxLife (Lu et al., 2024) and Bejjani et al. (2025) on emergent
behaviour in large evolved populations; W. Brian Arthur on technology as combination; Stuart
Kauffman on NK landscapes and the adjacent possible; Kenneth Stanley on novelty; Henrich on
cumulative culture; Lewis signalling games and Luc Steels' Talking Heads on emergent vocabulary;
and Dwarf Fortress and Noita on material-property-driven worlds.
