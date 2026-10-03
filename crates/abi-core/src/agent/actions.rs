//! The eight primitive actions. `decide` reads the frozen world; `apply_interior`
//! runs inside one chunk task and may touch only that chunk's cells and agents;
//! anything that crosses a chunk or mints a material is deferred to the
//! sequential boundary pass.

use super::brain::*;
use super::sensors::observe;
use super::*;
use crate::chem::generate::{MAT_CRUST, MAT_SOIL};
use crate::chem::props::*;
use crate::chem::Chemistry;
use crate::rng::{hash3, Rng};
use crate::stats::Stats;
use crate::world::cell::Cell;
use crate::world::grid::Grid;
use crate::world::World;

#[derive(Clone, Copy, Debug, PartialEq)]
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

impl Intent {
    pub fn code(&self) -> u8 {
        match self {
            Intent::Move(_) => 0,
            Intent::Take(_) => 1,
            Intent::Drop(_) => 2,
            Intent::Combine => 3,
            Intent::Heat => 4,
            Intent::Strike => 5,
            Intent::Give => 6,
            Intent::Emit(_) => 7,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Deferred {
    MoveAcross { agent_id: u64, to: (u16, u16) },
    Combine { agent_id: u64, a: MatId, b: MatId, temp: f32 },
}

pub struct ChunkCtx<'a> {
    pub cells: &'a mut [Cell],
    pub cell_base: usize,
    pub agents: &'a mut [Agent],
    pub agent_base_cell_start: &'a [u32],
    pub chem: &'a Chemistry,
    pub grid: &'a Grid,
    pub tick: u64,
    pub stats: &'a mut Stats,
    pub deferred: &'a mut Vec<Deferred>,
}

impl<'a> ChunkCtx<'a> {
    #[inline]
    fn cell_mut(&mut self, idx: usize) -> &mut Cell {
        &mut self.cells[idx - self.cell_base]
    }
    #[inline]
    fn cell(&self, idx: usize) -> &Cell {
        &self.cells[idx - self.cell_base]
    }
    /// Local indices (into `self.agents`) of agents in `cell`, in (cell, id) order.
    /// Membership is frozen at sort time: an interior `Move` mutates `x`/`y` in
    /// place, so callers must re-check position before treating a hit as present.
    fn locals_in(&self, cell: usize) -> std::ops::Range<usize> {
        let chunk_first = self.agent_base_cell_start[self.cell_base] as usize;
        let s = self.agent_base_cell_start[cell] as usize - chunk_first;
        let e = self.agent_base_cell_start[cell + 1] as usize - chunk_first;
        s..e
    }
}

pub fn decide(w: &World, a: &Agent) -> Intent {
    let mut rng = Rng::new(hash3(w.cfg.seed, w.tick, a.id));
    let mut input = [0f32; NIN];
    observe(w, a, &mut input);
    let mut out = [0f32; NOUT];
    forward(&a.genome, &input, &mut out);
    match sample(&out[O_ACT..O_ACT + NACT], &mut rng) {
        0 => Intent::Move(sample(&out[O_DIR..O_DIR + 4], &mut rng) as u8),
        1 => {
            let mut t = [0f32; NP];
            for i in 0..NP {
                t[i] = squash(out[O_TAKE + i]);
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

pub fn take_pick(cell: &Cell, chem: &Chemistry, target: &[f32; NP]) -> Option<MatId> {
    let dist = |id: MatId| -> f32 {
        let p = chem.props(id);
        (0..NP).map(|i| (p[i] - target[i]) * (p[i] - target[i])).sum()
    };
    cell.inv.iter().filter(|e| e.0 != MAT_SOIL && e.1 > 0).min_by(|p, q| dist(p.0).partial_cmp(&dist(q.0)).unwrap().then(p.0.cmp(&q.0))).map(|e| e.0)
}

pub fn apply_interior(ctx: &mut ChunkCtx, la: usize, intent: &Intent) {
    let code = intent.code() as usize;
    ctx.stats.actions[code] += 1;
    let (x, y, my_id) = {
        let a = &mut ctx.agents[la];
        a.energy -= ACTION_COST[code];
        a.last_action = code as u8;
        (a.x, a.y, a.id)
    };
    let here = ctx.grid.idx(x, y);
    match *intent {
        Intent::Move(d) => {
            let (nx, ny) = ctx.grid.step(x, y, d);
            let to = ctx.grid.idx(nx, ny);
            if ctx.grid.chunk_of(to) == ctx.grid.chunk_of(here) {
                let a = &mut ctx.agents[la];
                a.x = nx;
                a.y = ny;
            } else {
                ctx.deferred.push(Deferred::MoveAcross { agent_id: my_id, to: (nx, ny) });
            }
        }
        Intent::Take(target) => {
            let slot = match ctx.agents[la].free_slot() {
                Some(s) => s,
                None => return,
            };
            if let Some(id) = take_pick(ctx.cell(here), ctx.chem, &target) {
                let slot = ctx.agents[la].held.iter().position(|h| h.0 == id && h.1 > 0).unwrap_or(slot);
                let room = TAKE_MAX.saturating_sub(ctx.agents[la].held[slot].1);
                let got = ctx.cell_mut(here).remove(id, room);
                let a = &mut ctx.agents[la];
                a.held[slot] = (id, a.held[slot].1 + got);
            }
        }
        Intent::Drop(s) => {
            let (id, m) = ctx.agents[la].held[s as usize];
            ctx.agents[la].held[s as usize] = (0, 0);
            ctx.cell_mut(here).add(id, m);
        }
        Intent::Combine => {
            let [(ida, ma), (idb, mb)] = ctx.agents[la].held;
            if ma == 0 || mb == 0 {
                return;
            }
            let temp = ctx.cell(here).temp;
            ctx.deferred.push(Deferred::Combine { agent_id: my_id, a: ida, b: idb, temp });
        }
        Intent::Heat => {
            let c = ctx.cell_mut(here);
            c.temp = (c.temp + HEAT_DELTA).min(1.5);
        }
        Intent::Strike => {
            let (id, m) = ctx.agents[la].held[0];
            if m == 0 {
                return;
            }
            let hard = ctx.chem.props(id)[P_HARD];
            let crust_hard = ctx.chem.props(MAT_CRUST)[P_HARD];
            if ctx.cell(here).bedrock > 0 && hard > crust_hard {
                let c = ctx.cell_mut(here);
                let got = c.bedrock.min(STRIKE_YIELD);
                c.bedrock -= got;
                c.add(MAT_CRUST, got);
                ctx.stats.strikes_mine += 1;
                return;
            }
            let victim = ctx.locals_in(here).find(|&l| l != la && ctx.agents[l].alive && ctx.agents[l].x == x && ctx.agents[l].y == y);
            if let Some(v) = victim {
                let dmg = STRIKE_DAMAGE * hard * ctx.agents[la].genome.body_size;
                let vid = ctx.agents[v].id;
                ctx.agents[v].energy -= dmg;
                ctx.agents[v].memory.record(my_id, [0.0, 0.0, dmg, 0.0], ctx.tick as u32);
                ctx.agents[la].memory.record(vid, [0.0, 0.0, 0.0, 1.0], ctx.tick as u32);
                ctx.stats.strikes_hit += 1;
                return;
            }
            if let Some(r) = ctx.chem.table.recipe(id) {
                let half = m / 2;
                let a = &mut ctx.agents[la];
                if a.held[1].1 == 0 {
                    a.held = [(r.a, m - half), (r.b, half)];
                } else {
                    a.held[0] = (r.a, m - half);
                    ctx.cell_mut(here).add(r.b, half);
                }
                ctx.stats.strikes_split += 1;
            }
        }
        Intent::Give => {
            let (id, m) = ctx.agents[la].held[0];
            if m == 0 {
                return;
            }
            let target = ctx.locals_in(here).find(|&l| l != la && ctx.agents[l].alive && ctx.agents[l].x == x && ctx.agents[l].y == y && ctx.agents[l].free_slot().is_some());
            if let Some(t) = target {
                let tid = ctx.agents[t].id;
                let slot = ctx.agents[t].free_slot().unwrap();
                ctx.agents[t].held[slot] = (id, m);
                ctx.agents[la].held[0] = (0, 0);
                let value = ctx.chem.props(id)[P_NUTRI] * m as f32 / 1000.0 * ENERGY_PER_UNIT;
                ctx.agents[t].memory.record(my_id, [value, 0.0, 0.0, 0.0], ctx.tick as u32);
                ctx.agents[la].memory.record(tid, [0.0, value, 0.0, 0.0], ctx.tick as u32);
                ctx.stats.gives += 1;
            }
        }
        Intent::Emit(s) => {
            ctx.agents[la].signal = s;
        }
    }
}

/// Sequential boundary pass. `ai` is the index of the agent with `agent_id` in `w.agents`.
pub fn apply_deferred(w: &mut World, d: &Deferred, ai: usize) {
    match *d {
        Deferred::MoveAcross { to, .. } => {
            let a = &mut w.agents[ai];
            a.x = to.0;
            a.y = to.1;
        }
        Deferred::Combine { a, b, temp, .. } => {
            let [(_, ma), (_, mb)] = w.agents[ai].held;
            if ma == 0 || mb == 0 {
                return;
            }
            let id = w.chem.combine(a, b, temp);
            let ag = &mut w.agents[ai];
            ag.held = [(id, ma + mb), (0, 0)];
            w.stats.combines += 1;
            if Chemistry::hot(temp) {
                w.stats.combines_hot += 1;
            }
            w.stats.materials_in_use.push(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chem::generate::{ChemParams, MAT_AUTOTROPH, MAT_CRUST, MAT_SOIL};
    use crate::world::config::WorldConfig;
    use crate::world::World;

    fn world(pop: usize) -> World {
        World::new(&WorldConfig { seed: 4, width: 32, height: 32, pop0: pop, chem: ChemParams { n_base: 12, ..Default::default() }, ..Default::default() })
    }

    /// Run `f` as if it were the chunk task for agent 0's chunk.
    fn with_ctx<R>(w: &mut World, f: impl FnOnce(&mut ChunkCtx) -> R) -> R {
        let cell = w.grid.idx(w.agents[0].x, w.agents[0].y);
        let ch = w.grid.chunk_of(cell);
        let cell_base = ch * World::CHUNK_CELLS;
        let (a0, a1) = (w.agent_start[ch] as usize, w.agent_start[ch + 1] as usize);
        let mut deferred = Vec::new();
        let mut stats = Stats::default();
        let World { grid, agents, chem, cell_start, tick, .. } = w;
        let cells = &mut grid.cells[cell_base..cell_base + World::CHUNK_CELLS];
        let grid_ro = Grid { width: grid.width, height: grid.height, cells: Vec::new() };
        let mut ctx = ChunkCtx { cells, cell_base, agents: &mut agents[a0..a1], agent_base_cell_start: &cell_start[..], chem, grid: &grid_ro, tick: *tick, stats: &mut stats, deferred: &mut deferred };
        f(&mut ctx)
    }

    #[test]
    fn take_picks_nearest_in_property_space_and_ignores_soil() {
        let mut w = world(1);
        let cell = w.grid.idx(w.agents[0].x, w.agents[0].y);
        w.grid.cells[cell].inv.clear();
        w.grid.cells[cell].add(MAT_SOIL, 5000);
        w.grid.cells[cell].add(MAT_AUTOTROPH, 500);
        w.grid.cells[cell].add(MAT_CRUST, 500);
        let crust = *w.chem.props(MAT_CRUST);
        assert_eq!(take_pick(&w.grid.cells[cell], &w.chem, &crust), Some(MAT_CRUST));
        let only_soil = { let mut c = Cell::default(); c.add(MAT_SOIL, 100); c };
        assert_eq!(take_pick(&only_soil, &w.chem, &crust), None);
    }

    #[test]
    fn take_moves_mass_into_a_slot_and_conserves() {
        let mut w = world(1);
        let cell = w.grid.idx(w.agents[0].x, w.agents[0].y);
        w.grid.cells[cell].add(MAT_AUTOTROPH, 2500);
        let before = w.total_mass();
        let target = *w.chem.props(MAT_AUTOTROPH);
        with_ctx(&mut w, |ctx| {
            let local = ctx.agents.iter().position(|a| a.id == 0).unwrap();
            apply_interior(ctx, local, &Intent::Take(target));
        });
        assert_eq!(w.agents[0].held[0], (MAT_AUTOTROPH, TAKE_MAX));
        assert_eq!(w.total_mass(), before);
        assert_eq!(w.agents[0].last_action, 1);
    }

    #[test]
    fn combine_is_deferred_and_same_material_self_reacts() {
        let mut w = world(1);
        w.agents[0].held = [(MAT_AUTOTROPH, 300), (MAT_AUTOTROPH, 200)];
        let before = w.total_mass();
        let deferred = with_ctx(&mut w, |ctx| {
            let local = ctx.agents.iter().position(|a| a.id == 0).unwrap();
            apply_interior(ctx, local, &Intent::Combine);
            ctx.deferred.clone()
        });
        assert_eq!(deferred.len(), 1);
        apply_deferred(&mut w, &deferred[0], 0);
        let (id, m) = w.agents[0].held[0];
        assert_eq!(m, 500);
        assert_eq!(w.agents[0].held[1], (0, 0));
        assert_eq!(w.chem.table.recipe(id).map(|r| (r.a, r.b)), Some((MAT_AUTOTROPH, MAT_AUTOTROPH)));
        assert_eq!(w.total_mass(), before);
    }

    #[test]
    fn strike_precedence_mine_then_hit_then_split() {
        let mut w = world(2);
        // Put both agents on one bedrock cell; agent 0 holds something harder than crust.
        let (x, y) = (w.agents[0].x, w.agents[0].y);
        w.agents[1].x = x;
        w.agents[1].y = y;
        w.sort_agents();
        let cell = w.grid.idx(x, y);
        w.grid.cells[cell].bedrock = 10_000;
        let hard_raw = { let mut p = [0f32; NP]; p[P_HARD] = 5.0; p };
        let hard = w.chem.table.intern(hard_raw, crate::chem::table::Recipe { a: 1, b: 2, tq: 0 });
        let me = w.agents.iter().position(|a| a.id == 0).unwrap();
        w.agents[me].held[0] = (hard, 100);
        let before = w.total_mass();
        with_ctx(&mut w, |ctx| {
            let local = ctx.agents.iter().position(|a| a.id == 0).unwrap();
            apply_interior(ctx, local, &Intent::Strike);
        });
        assert_eq!(w.grid.cells[cell].bedrock, 10_000 - STRIKE_YIELD);
        assert_eq!(w.grid.cells[cell].get(MAT_CRUST), STRIKE_YIELD);
        assert_eq!(w.total_mass(), before);
        // No bedrock: hit the other agent.
        w.grid.cells[cell].bedrock = 0;
        let other = w.agents.iter().position(|a| a.id == 1).unwrap();
        let e_before = w.agents[other].energy;
        with_ctx(&mut w, |ctx| {
            let local = ctx.agents.iter().position(|a| a.id == 0).unwrap();
            apply_interior(ctx, local, &Intent::Strike);
        });
        assert!(w.agents[other].energy < e_before);
        assert!(w.agents[other].memory.knows(0), "victim remembers the striker");
        // Alone with an artifact: split.
        let victim = w.agents.iter().position(|a| a.id == 1).unwrap();
        w.agents[victim].alive = false;
        w.sort_agents();
        let me = w.agents.iter().position(|a| a.id == 0).unwrap();
        w.agents[me].held = [(hard, 100), (0, 0)];
        with_ctx(&mut w, |ctx| {
            let local = ctx.agents.iter().position(|a| a.id == 0).unwrap();
            apply_interior(ctx, local, &Intent::Strike);
        });
        assert_eq!(w.agents[me].held, [(1, 50), (2, 50)]);
    }

    #[test]
    fn give_transfers_slot_and_both_remember() {
        let mut w = world(2);
        let (x, y) = (w.agents[0].x, w.agents[0].y);
        w.agents[1].x = x;
        w.agents[1].y = y;
        w.sort_agents();
        // After sorting, index 0 is not necessarily id 0: address the giver by id.
        let giver_ix = w.agents.iter().position(|a| a.id == 0).unwrap();
        w.agents[giver_ix].held[0] = (MAT_AUTOTROPH, 400);
        with_ctx(&mut w, |ctx| {
            let local = ctx.agents.iter().position(|a| a.id == 0).unwrap();
            apply_interior(ctx, local, &Intent::Give);
        });
        let giver = w.agents.iter().find(|a| a.id == 0).unwrap();
        let taker = w.agents.iter().find(|a| a.id == 1).unwrap();
        assert_eq!(giver.held[0], (0, 0));
        assert!(taker.held.contains(&(MAT_AUTOTROPH, 400)));
        assert!(giver.memory.knows(1) && taker.memory.knows(0));
    }

    #[test]
    fn move_across_chunk_is_deferred() {
        let mut w = World::new(&WorldConfig { seed: 4, width: 64, height: 32, pop0: 1, ..Default::default() });
        w.agents[0].x = 31;
        w.agents[0].y = 5;
        w.sort_agents();
        let deferred = with_ctx(&mut w, |ctx| {
            let local = 0;
            apply_interior(ctx, local, &Intent::Move(0));
            ctx.deferred.clone()
        });
        assert_eq!(w.agents[0].x, 31, "not moved yet");
        assert!(matches!(deferred[0], Deferred::MoveAcross { to: (32, 5), .. }));
        apply_deferred(&mut w, &deferred[0], 0);
        assert_eq!((w.agents[0].x, w.agents[0].y), (32, 5));
    }

    #[test]
    fn strike_and_give_skip_agents_that_moved_away() {
        let mut w = World::new(&WorldConfig { seed: 4, width: 64, height: 32, pop0: 2, ..Default::default() });
        // x = 5: the +x neighbour (x = 6) is in the same chunk, so Move is interior.
        for a in w.agents.iter_mut() {
            a.x = 5;
            a.y = 5;
        }
        w.sort_agents();
        let cell = w.grid.idx(5, 5);
        w.grid.cells[cell].bedrock = 0;
        let ix = |w: &World, id: u64| w.agents.iter().position(|a| a.id == id).unwrap();
        let (i0, i1) = (ix(&w, 0), ix(&w, 1));
        w.agents[i0].held = [(MAT_AUTOTROPH, 100), (0, 0)];
        w.agents[i1].held = [(0, 0), (0, 0)];
        let e1 = with_ctx(&mut w, |ctx| {
            let l0 = ctx.agents.iter().position(|a| a.id == 0).unwrap();
            let l1 = ctx.agents.iter().position(|a| a.id == 1).unwrap();
            apply_interior(ctx, l1, &Intent::Move(0));
            assert_eq!(ctx.agents[l1].x, 6, "interior move happened");
            assert!(ctx.deferred.is_empty());
            let e1 = ctx.agents[l1].energy;
            apply_interior(ctx, l0, &Intent::Strike);
            apply_interior(ctx, l0, &Intent::Give);
            e1
        });
        let (a0, a1) = (&w.agents[i0], &w.agents[i1]);
        assert_eq!(a1.energy, e1, "moved-away agent was not hit");
        assert_eq!(a1.held, [(0, 0), (0, 0)], "moved-away agent received nothing");
        assert_eq!(a0.held[0], (MAT_AUTOTROPH, 100), "giver kept its item");
        assert!(!a0.memory.knows(1) && !a1.memory.knows(0));
    }

    #[test]
    fn decide_is_deterministic_for_same_tick() {
        let w = world(5);
        let a = &w.agents[0];
        let i1 = decide(&w, a);
        let i2 = decide(&w, a);
        assert_eq!(i1.code(), i2.code());
    }
}
