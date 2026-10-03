//! Headless grid world. Two-phase deterministic tick:
//!   phase 1 (parallel): every agent reads the frozen world and emits an intent;
//!   phase 2 (sequential): intents are applied in (cell, agent id) order.
//! Mass is integer and exactly conserved. Energy is open (star in, heat out).

use crate::brain::*;
use crate::chem::*;
use crate::rng::{hash3, Rng};
use rayon::prelude::*;
use std::collections::HashSet;

pub const BODY_TARGET: u32 = 500;
pub const BODY_REPRO: u32 = 1000;
pub const TAKE_MAX: u32 = 1000;
pub const DIGEST_RATE: u32 = 100;
pub const ENERGY_PER_UNIT: f32 = 150.0; // energy from 1000 mass at nutrition 1.0
pub const BASE_COST: f32 = 1.0;
pub const REPRO_ENERGY: f32 = 250.0;
pub const START_ENERGY: f32 = 200.0;
pub const LIFESPAN: u32 = 1500;
pub const AMBIENT: f32 = 0.2;
pub const HEAT_DELTA: f32 = 0.2;
pub const TEMP_DECAY: f32 = 0.85;
pub const PLANT_GROWTH: u32 = 40;
pub const PLANT_CAP: u32 = 3000;
pub const STRIKE_YIELD: u32 = 500;
pub const SOIL_DIFFUSION: u32 = 8;

const COST: [f32; 8] = [0.3, 0.2, 0.2, 1.0, 3.0, 2.0, 0.2, 0.1];
pub const ACTION_NAMES: [&str; 8] = ["move", "take", "drop", "combine", "heat", "strike", "give", "emit"];

#[derive(Clone, Default)]
pub struct Cell {
    pub inv: Vec<(u32, u32)>, // (material id, mass)
    pub temp: f32,
    pub bedrock: u32, // mass of MAT_ROCK locked in the crust
    pub fertile: bool,
}

impl Cell {
    fn add(&mut self, id: u32, mass: u32) {
        if mass == 0 {
            return;
        }
        if let Some(e) = self.inv.iter_mut().find(|e| e.0 == id) {
            e.1 += mass;
        } else {
            self.inv.push((id, mass));
        }
    }
    fn remove(&mut self, id: u32, mass: u32) -> u32 {
        if let Some(i) = self.inv.iter().position(|e| e.0 == id) {
            let take = self.inv[i].1.min(mass);
            self.inv[i].1 -= take;
            if self.inv[i].1 == 0 {
                self.inv.swap_remove(i);
            }
            take
        } else {
            0
        }
    }
    fn get(&self, id: u32) -> u32 {
        self.inv.iter().find(|e| e.0 == id).map(|e| e.1).unwrap_or(0)
    }
    fn mass(&self) -> u64 {
        self.inv.iter().map(|e| e.1 as u64).sum::<u64>() + self.bedrock as u64
    }
}

#[derive(Clone)]
pub struct Agent {
    pub alive: bool,
    pub id: u64,
    pub x: u16,
    pub y: u16,
    pub energy: f32,
    pub age: u32,
    pub body: u32,
    pub held: [(u32, u32); 2],
    pub signal: [f32; 2],
    pub genome: Genome,
}

#[derive(Clone, Copy, Debug)]
pub enum Intent {
    Move(u8),
    Take([f32; NP]),
    Drop(u8),
    Combine,
    Heat,
    Strike,
    Give,
    Emit([f32; 2]),
}

#[derive(Default, Clone)]
pub struct Stats {
    pub agent_steps: u64,
    pub actions: [u64; 8],
    pub combines_ok: u64,
    pub combines_hot: u64,
    pub strikes_ok: u64,
    pub gives_ok: u64,
    pub births: u64,
    pub deaths: u64,
    pub energy_total: f64,
    pub energy_from_artifacts: f64,
    pub artifacts_in_use: HashSet<u32>,
    pub materials_eaten: HashSet<u32>,
}

pub struct World {
    pub seed: u64,
    pub w: usize,
    pub h: usize,
    pub tick: u64,
    pub cells: Vec<Cell>,
    pub agents: Vec<Agent>,
    pub chem: Chemistry,
    pub next_id: u64,
    pub max_agents: usize,
    pub random_policy: bool,
    pub stats: Stats,
    order: Vec<u32>, // agent indices sorted by (cell, id)
    cell_start: Vec<u32>,
    free: Vec<u32>,
}

impl World {
    pub fn new(seed: u64, w: usize, h: usize, n_base: usize, pop0: usize, max_agents: usize, random_policy: bool) -> Self {
        let chem = Chemistry::generate(seed, n_base);
        let mut rng = Rng::new(seed ^ 0x3031_4C44_0000_0001);
        let mut cells = vec![Cell::default(); w * h];
        for c in cells.iter_mut() {
            c.temp = AMBIENT;
            c.add(MAT_SOIL, 5000);
            c.fertile = rng.f32() < 0.5;
            if c.fertile {
                c.add(MAT_PLANT, 1000);
            }
            if rng.f32() < 0.3 {
                c.bedrock = 20000;
            }
            for m in 3..n_base as u32 {
                if rng.f32() < 0.03 {
                    c.add(m, 2000);
                }
            }
        }
        let mut agents = Vec::with_capacity(max_agents);
        for i in 0..pop0 {
            let mut r = Rng::new(hash3(seed, 0xA6E7, i as u64));
            let x = r.range(w) as u16;
            let y = r.range(h) as u16;
            let genome = Genome::random(&mut r);
            // Agent body mass is taken from the soil of its starting cell (closed matter).
            let cell = &mut cells[y as usize * w + x as usize];
            let got = cell.remove(MAT_SOIL, BODY_TARGET);
            agents.push(Agent { alive: true, id: i as u64, x, y, energy: START_ENERGY, age: 0, body: got, held: [(0, 0); 2], signal: [0.0; 2], genome });
        }
        World { seed, w, h, tick: 0, cells, agents, chem, next_id: pop0 as u64, max_agents, random_policy, stats: Stats::default(), order: Vec::new(), cell_start: vec![0; w * h + 1], free: Vec::new() }
    }

    #[inline]
    fn cidx(&self, x: u16, y: u16) -> usize {
        y as usize * self.w + x as usize
    }

    #[inline]
    fn step_dir(&self, x: u16, y: u16, d: u8) -> (u16, u16) {
        let (w, h) = (self.w as i32, self.h as i32);
        let (dx, dy) = [(1, 0), (-1, 0), (0, 1), (0, -1)][d as usize & 3];
        ((((x as i32 + dx) % w + w) % w) as u16, (((y as i32 + dy) % h + h) % h) as u16)
    }

    /// Sort living agents by (cell, id); build per-cell ranges. Serves both
    /// neighbour lookup in phase 1 and application order in phase 2.
    fn build_order(&mut self) {
        self.order.clear();
        for (i, a) in self.agents.iter().enumerate() {
            if a.alive {
                self.order.push(i as u32);
            }
        }
        let (w, agents) = (self.w, &self.agents);
        self.order.sort_unstable_by_key(|&i| {
            let a = &agents[i as usize];
            ((a.y as usize * w + a.x as usize) as u64, a.id)
        });
        self.cell_start.fill(0);
        for &i in &self.order {
            let a = &self.agents[i as usize];
            let c = a.y as usize * self.w + a.x as usize;
            self.cell_start[c + 1] += 1;
        }
        for c in 0..self.w * self.h {
            self.cell_start[c + 1] += self.cell_start[c];
        }
    }

    fn agents_in(&self, c: usize) -> &[u32] {
        &self.order[self.cell_start[c] as usize..self.cell_start[c + 1] as usize]
    }

    fn observe(&self, a: &Agent, input: &mut [f32; NIN]) {
        let mut k = 0;
        input[k] = a.energy / 300.0;
        k += 1;
        input[k] = a.age as f32 / LIFESPAN as f32;
        k += 1;
        for s in 0..2 {
            let (id, mass) = a.held[s];
            if mass > 0 {
                let p = &self.chem.props[id as usize];
                input[k..k + NP].copy_from_slice(p);
            } else {
                input[k..k + NP].fill(0.0);
            }
            input[k + NP] = mass as f32 / TAKE_MAX as f32;
            k += NP + 1;
        }
        // Here + 4 neighbours: max nutrition, max hardness, total loose mass, temp.
        let here = self.cidx(a.x, a.y);
        let mut cells = [here; 5];
        for d in 0..4u8 {
            let (x, y) = self.step_dir(a.x, a.y, d);
            cells[d as usize + 1] = self.cidx(x, y);
        }
        for &c in &cells {
            let cell = &self.cells[c];
            let (mut mn, mut mh, mut tot) = (0f32, 0f32, 0u32);
            for &(id, m) in &cell.inv {
                let p = &self.chem.props[id as usize];
                mn = mn.max(p[P_NUTRI]);
                mh = mh.max(p[P_HARD]);
                tot += m;
            }
            input[k] = mn;
            input[k + 1] = mh;
            input[k + 2] = (tot as f32 / 5000.0).min(2.0);
            input[k + 3] = cell.temp;
            k += 4;
        }
        // Properties of the two most massive loose items here (soil excluded).
        {
            let mut items: Vec<(u32, u32)> = self.cells[here].inv.iter().copied().filter(|e| e.0 != MAT_SOIL).collect();
            items.sort_by(|p, q| q.1.cmp(&p.1).then(p.0.cmp(&q.0)));
            for s in 0..2 {
                if let Some(&(id, _)) = items.get(s) {
                    input[k..k + NP].copy_from_slice(&self.chem.props[id as usize]);
                } else {
                    input[k..k + NP].fill(0.0);
                }
                k += NP;
            }
        }
        input[k] = if self.cells[here].bedrock > 0 { self.chem.props[MAT_ROCK as usize][P_HARD] } else { 0.0 };
        k += 1;
        let others = self.agents_in(here);
        input[k] = (others.len() as f32 - 1.0).max(0.0) / 5.0;
        k += 1;
        // Signal of the lowest-id other agent in the cell.
        let mut sig = [0f32; 2];
        for &o in others {
            let o = &self.agents[o as usize];
            if o.id != a.id {
                sig = o.signal;
                break;
            }
        }
        input[k] = sig[0];
        input[k + 1] = sig[1];
        k += 2;
        debug_assert_eq!(k, NIN);
    }

    fn decide(&self, ai: u32) -> Intent {
        let a = &self.agents[ai as usize];
        let mut rng = Rng::new(hash3(self.seed, self.tick, a.id));
        let mut out = [0f32; NOUT];
        if self.random_policy {
            for o in out.iter_mut() {
                *o = rng.normal();
            }
        } else {
            let mut input = [0f32; NIN];
            self.observe(a, &mut input);
            a.genome.forward(&input, &mut out);
        }
        match sample(&out[O_ACT..O_ACT + 8], &mut rng) {
            0 => Intent::Move(sample(&out[O_DIR..O_DIR + 4], &mut rng) as u8),
            1 => {
                let mut t = [0f32; NP];
                for i in 0..NP {
                    t[i] = 1.0 / (1.0 + (-out[O_TAKE + i]).exp());
                }
                Intent::Take(t)
            }
            2 => Intent::Drop(sample(&out[O_DROP..O_DROP + 2], &mut rng) as u8),
            3 => Intent::Combine,
            4 => Intent::Heat,
            5 => Intent::Strike,
            6 => Intent::Give,
            _ => Intent::Emit([out[O_EMIT].tanh(), out[O_EMIT + 1].tanh()]),
        }
    }

    fn apply(&mut self, ai: u32, intent: Intent) {
        let (x, y) = {
            let a = &self.agents[ai as usize];
            (a.x, a.y)
        };
        let here = self.cidx(x, y);
        let code = match intent {
            Intent::Move(_) => 0,
            Intent::Take(_) => 1,
            Intent::Drop(_) => 2,
            Intent::Combine => 3,
            Intent::Heat => 4,
            Intent::Strike => 5,
            Intent::Give => 6,
            Intent::Emit(_) => 7,
        };
        self.stats.actions[code] += 1;
        self.agents[ai as usize].energy -= COST[code];
        match intent {
            Intent::Move(d) => {
                let (nx, ny) = self.step_dir(x, y, d);
                let a = &mut self.agents[ai as usize];
                a.x = nx;
                a.y = ny;
            }
            Intent::Take(target) => {
                let slot = match self.agents[ai as usize].held.iter().position(|h| h.1 == 0) {
                    Some(s) => s,
                    None => return,
                };
                let dist = |id: u32| -> f32 {
                    let p = &self.chem.props[id as usize];
                    (0..NP).map(|i| (p[i] - target[i]) * (p[i] - target[i])).sum()
                };
                let pick = self.cells[here].inv.iter().filter(|e| e.0 != MAT_SOIL).min_by(|p, q| {
                    dist(p.0).partial_cmp(&dist(q.0)).unwrap().then(p.0.cmp(&q.0))
                });
                if let Some(&(id, _)) = pick {
                    // Stack onto an existing slot of the same id, else the empty slot.
                    let a = &mut self.agents[ai as usize];
                    let slot = a.held.iter().position(|h| h.0 == id && h.1 > 0).unwrap_or(slot);
                    let room = TAKE_MAX.saturating_sub(a.held[slot].1);
                    let got = self.cells[here].remove(id, room);
                    let a = &mut self.agents[ai as usize];
                    a.held[slot] = (id, a.held[slot].1 + got);
                }
            }
            Intent::Drop(s) => {
                let a = &mut self.agents[ai as usize];
                let (id, m) = a.held[s as usize];
                a.held[s as usize] = (0, 0);
                self.cells[here].add(id, m);
            }
            Intent::Combine => {
                let a = &self.agents[ai as usize];
                let [(ida, ma), (idb, mb)] = a.held;
                if ma == 0 || mb == 0 {
                    return;
                }
                let temp = self.cells[here].temp;
                let id = self.chem.combine(ida, idb, temp);
                let hot = temp >= Chemistry::hot_threshold(&self.chem.props[ida as usize], &self.chem.props[idb as usize]);
                let a = &mut self.agents[ai as usize];
                a.held = [(id, ma + mb), (0, 0)];
                self.stats.combines_ok += 1;
                if hot {
                    self.stats.combines_hot += 1;
                }
                self.stats.artifacts_in_use.insert(id);
            }
            Intent::Heat => {
                self.cells[here].temp = (self.cells[here].temp + HEAT_DELTA).min(1.5);
            }
            Intent::Strike => {
                let (id, m) = self.agents[ai as usize].held[0];
                if m == 0 {
                    return;
                }
                let hard = self.chem.props[id as usize][P_HARD];
                let rock_hard = self.chem.props[MAT_ROCK as usize][P_HARD];
                if self.cells[here].bedrock > 0 && hard > rock_hard {
                    let got = self.cells[here].bedrock.min(STRIKE_YIELD);
                    self.cells[here].bedrock -= got;
                    self.cells[here].add(MAT_ROCK, got);
                    self.stats.strikes_ok += 1;
                } else if let Some(r) = self.chem.recipe[id as usize] {
                    // Break an artifact back into its parts (mass split, exact).
                    let half = m / 2;
                    let a = &mut self.agents[ai as usize];
                    if a.held[1].1 == 0 {
                        a.held = [(r.a, m - half), (r.b, half)];
                    } else {
                        a.held[0] = (r.a, m - half);
                        self.cells[here].add(r.b, half);
                    }
                    self.stats.strikes_ok += 1;
                }
            }
            Intent::Give => {
                let (id, m) = self.agents[ai as usize].held[0];
                if m == 0 {
                    return;
                }
                let my_id = self.agents[ai as usize].id;
                let target = self.agents_in(here).iter().copied().find(|&o| {
                    let o = &self.agents[o as usize];
                    o.alive && o.id != my_id && o.held.iter().any(|h| h.1 == 0)
                });
                if let Some(t) = target {
                    let t = &mut self.agents[t as usize];
                    let slot = t.held.iter().position(|h| h.1 == 0).unwrap();
                    t.held[slot] = (id, m);
                    self.agents[ai as usize].held[0] = (0, 0);
                    self.stats.gives_ok += 1;
                }
            }
            Intent::Emit(s) => {
                self.agents[ai as usize].signal = s;
            }
        }
    }

    /// Metabolism, digestion, excretion, death, reproduction. Sequential.
    fn metabolize(&mut self) {
        let n = self.agents.len();
        for i in 0..n {
            if !self.agents[i].alive {
                continue;
            }
            let (x, y, id) = (self.agents[i].x, self.agents[i].y, self.agents[i].id);
            let here = self.cidx(x, y);
            let a = &mut self.agents[i];
            a.energy -= BASE_COST;
            a.age += 1;
            // Digestion: anything held above the digestion threshold is food.
            for s in 0..2 {
                let (mid, m) = a.held[s];
                if m == 0 {
                    continue;
                }
                let p = &self.chem.props[mid as usize];
                if p[P_NUTRI] > a.genome.digest_thr {
                    let d = m.min(DIGEST_RATE);
                    let gain = (p[P_NUTRI] - 1.5 * p[P_TOXIC]) * d as f32 / 1000.0 * ENERGY_PER_UNIT;
                    a.energy += gain;
                    a.body += d;
                    a.held[s].1 -= d;
                    if a.held[s].1 == 0 {
                        a.held[s] = (0, 0);
                    }
                    self.stats.energy_total += gain.max(0.0) as f64;
                    self.stats.materials_eaten.insert(mid);
                    if self.chem.is_artifact(mid) {
                        self.stats.energy_from_artifacts += gain.max(0.0) as f64;
                        self.stats.artifacts_in_use.insert(mid);
                    }
                }
            }
            // Excrete above body target (unless saving for reproduction).
            if a.body > BODY_REPRO + 200 {
                let ex = a.body - (BODY_REPRO + 200);
                a.body -= ex;
                self.cells[here].add(MAT_SOIL, ex);
            }
            // Death: everything returns to the cell.
            if a.energy <= 0.0 || a.age > LIFESPAN {
                a.alive = false;
                let body = a.body;
                let held = a.held;
                a.body = 0;
                a.held = [(0, 0); 2];
                self.cells[here].add(MAT_SOIL, body);
                for (hid, hm) in held {
                    self.cells[here].add(hid, hm);
                }
                self.stats.deaths += 1;
                self.free.push(i as u32);
                continue;
            }
            // Reproduction: asexual division.
            if a.energy >= REPRO_ENERGY && a.body >= BODY_REPRO {
                let slot = if let Some(f) = self.free.pop() {
                    Some(f as usize)
                } else if self.agents.len() < self.max_agents {
                    None
                } else {
                    continue;
                };
                let a = &mut self.agents[i];
                a.energy *= 0.5;
                a.body -= BODY_TARGET;
                let child_energy = a.energy;
                let mut rng = Rng::new(hash3(self.seed ^ 0xB1B7, self.tick, id));
                let genome = a.genome.mutate(&mut rng);
                let d = rng.range(4) as u8;
                let (cx, cy) = self.step_dir(x, y, d);
                let child = Agent { alive: true, id: self.next_id, x: cx, y: cy, energy: child_energy, age: 0, body: BODY_TARGET, held: [(0, 0); 2], signal: [0.0; 2], genome };
                self.next_id += 1;
                self.stats.births += 1;
                match slot {
                    Some(s) => self.agents[s] = child,
                    None => self.agents.push(child),
                }
            }
        }
    }

    fn physics(&mut self) {
        // Soil diffuses to neighbours (exact integer transfers). Without circulation,
        // matter locks up in sinks and the biosphere starves.
        let (w, h) = (self.w, self.h);
        for y in 0..h {
            for x in 0..w {
                let c = y * w + x;
                for n in [y * w + (x + 1) % w, ((y + 1) % h) * w + x] {
                    let (sc, sn) = (self.cells[c].get(MAT_SOIL), self.cells[n].get(MAT_SOIL));
                    if sc > sn {
                        let f = (sc - sn) / SOIL_DIFFUSION;
                        let got = self.cells[c].remove(MAT_SOIL, f);
                        self.cells[n].add(MAT_SOIL, got);
                    } else {
                        let f = (sn - sc) / SOIL_DIFFUSION;
                        let got = self.cells[n].remove(MAT_SOIL, f);
                        self.cells[c].add(MAT_SOIL, got);
                    }
                }
            }
        }
        for c in self.cells.iter_mut() {
            c.temp = AMBIENT + (c.temp - AMBIENT) * TEMP_DECAY;
            if c.fertile {
                let plant = c.inv.iter().find(|e| e.0 == MAT_PLANT).map(|e| e.1).unwrap_or(0);
                if plant < PLANT_CAP {
                    let g = c.remove(MAT_SOIL, PLANT_GROWTH.min(PLANT_CAP - plant));
                    c.add(MAT_PLANT, g);
                }
            }
        }
    }

    pub fn step(&mut self) {
        self.build_order();
        let intents: Vec<Intent> = self.order.par_iter().map(|&i| self.decide(i)).collect();
        self.stats.agent_steps += intents.len() as u64;
        for k in 0..self.order.len() {
            let i = self.order[k];
            self.apply(i, intents[k]);
        }
        self.metabolize();
        self.physics();
        self.tick += 1;
    }

    pub fn population(&self) -> usize {
        self.agents.iter().filter(|a| a.alive).count()
    }

    /// Total mass anywhere: cells (loose + bedrock) + agent bodies + held.
    pub fn total_mass(&self) -> u64 {
        let cells: u64 = self.cells.iter().map(|c| c.mass()).sum();
        let agents: u64 = self.agents.iter().filter(|a| a.alive).map(|a| a.body as u64 + a.held[0].1 as u64 + a.held[1].1 as u64).sum();
        cells + agents
    }

    pub fn state_hash(&self) -> u64 {
        let mut h = 0x1234_5678_9abc_def0u64;
        let mut mix = |v: u64| h = crate::rng::mix64(h ^ v.wrapping_add(0x9e37_79b9_7f4a_7c15));
        for a in &self.agents {
            if a.alive {
                mix(a.id);
                mix(a.x as u64 | (a.y as u64) << 16);
                mix(a.energy.to_bits() as u64);
                mix(a.body as u64);
                mix(a.held[0].0 as u64 | (a.held[0].1 as u64) << 32);
                mix(a.held[1].0 as u64 | (a.held[1].1 as u64) << 32);
            }
        }
        for c in &self.cells {
            let mut inv = c.inv.clone();
            inv.sort();
            for (id, m) in inv {
                mix(id as u64 | (m as u64) << 32);
            }
            mix(c.temp.to_bits() as u64);
            mix(c.bedrock as u64);
        }
        mix(self.chem.n_known() as u64);
        h
    }

    pub fn take_stats(&mut self) -> Stats {
        std::mem::take(&mut self.stats)
    }

    pub fn soil_split(&self) -> (u64, u64, u64) {
        let mut fert = 0u64;
        let mut infert = 0u64;
        let mut plant = 0u64;
        for c in &self.cells {
            if c.fertile { fert += c.get(MAT_SOIL) as u64 } else { infert += c.get(MAT_SOIL) as u64 }
            plant += c.get(MAT_PLANT) as u64;
        }
        (fert, infert, plant)
    }

    pub fn held_artifacts(&self) -> usize {
        let mut s = HashSet::new();
        for a in &self.agents {
            if a.alive {
                for h in a.held {
                    if h.1 > 0 && self.chem.is_artifact(h.0) {
                        s.insert(h.0);
                    }
                }
            }
        }
        s.len()
    }
}
