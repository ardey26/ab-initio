//! The tick. Parallel over agents (decide) and over chunks (apply, metabolize,
//! physics); sequential boundary pass in chunk order; parallel field passes.

use super::fields::{diffuse_soil, diffuse_water, relax_temperature};
use super::grid::Grid;
use super::physics::{chunk_physics, mint_reactions, PendingReaction, TEMP_DECAY};
use super::World;
use crate::agent::actions::{apply_deferred, apply_interior, decide, ChunkCtx, Deferred, Intent};
use crate::agent::metabolism::metabolize_chunk;
use crate::agent::Agent;
use crate::stats::Stats;
use crate::world::cell::Cell;
use rayon::prelude::*;
use std::collections::HashMap;

struct ChunkOut {
    stats: Stats,
    deferred: Vec<Deferred>,
    births: Vec<Agent>,
    pending: Vec<PendingReaction>,
}

/// Split cells into fixed-size chunk slices and agents into the per-chunk ranges.
pub fn split_chunks<'a>(cells: &'a mut [Cell], agents: &'a mut [Agent], agent_start: &[u32]) -> Vec<(&'a mut [Cell], &'a mut [Agent])> {
    let mut out = Vec::with_capacity(agent_start.len() - 1);
    let mut rest_agents = agents;
    for (ch, cs) in cells.chunks_mut(World::CHUNK_CELLS).enumerate() {
        let n = (agent_start[ch + 1] - agent_start[ch]) as usize;
        let (mine, rest) = rest_agents.split_at_mut(n);
        rest_agents = rest;
        out.push((cs, mine));
    }
    debug_assert!(rest_agents.is_empty(), "agents not fully assigned to chunks; was sort_agents called?");
    out
}

impl World {
    pub fn step(&mut self) {
        self.sort_agents();
        // 2. decide (parallel over agents, frozen world)
        let intents: Vec<Intent> = {
            let w = &*self;
            w.agents.par_iter().map(|a| decide(w, a)).collect()
        };
        self.stats.agent_steps += intents.len() as u64;
        // 3. chunk tasks
        let grid_ro = Grid { width: self.grid.width, height: self.grid.height, cells: Vec::new() };
        let (seed, tick, growth) = (self.cfg.seed, self.tick, self.growth);
        let chem = &self.chem;
        let cell_start = &self.cell_start;
        let agent_start = self.agent_start.clone();
        let outs: Vec<ChunkOut> = {
            let parts = split_chunks(&mut self.grid.cells, &mut self.agents, &agent_start);
            parts
                .into_par_iter()
                .enumerate()
                .map(|(ch, (cells, agents))| {
                    let cell_base = ch * World::CHUNK_CELLS;
                    let a0 = agent_start[ch] as usize;
                    let mut out = ChunkOut { stats: Stats::default(), deferred: Vec::new(), births: Vec::new(), pending: Vec::new() };
                    {
                        let mut ctx = ChunkCtx { cells, cell_base, agents, agent_base_cell_start: cell_start, chem, grid: &grid_ro, tick, stats: &mut out.stats, deferred: &mut out.deferred };
                        for la in 0..ctx.agents.len() {
                            apply_interior(&mut ctx, la, &intents[a0 + la]);
                        }
                        metabolize_chunk(&mut ctx, seed, &mut out.births);
                        let ChunkCtx { cells, stats, .. } = ctx;
                        chunk_physics(cells, cell_base, seed, tick, chem, growth, stats, &mut out.pending);
                    }
                    out
                })
                .collect()
        };
        // 4. sequential boundary pass in chunk order
        let index: HashMap<u64, usize> = self.agents.iter().enumerate().map(|(i, a)| (a.id, i)).collect();
        for out in &outs {
            self.stats.merge(&out.stats);
            for d in &out.deferred {
                let id = match d {
                    Deferred::MoveAcross { agent_id, .. } | Deferred::Combine { agent_id, .. } => *agent_id,
                };
                if let Some(&ai) = index.get(&id) {
                    if self.agents[ai].alive {
                        apply_deferred(self, d, ai);
                    }
                }
            }
        }
        let mut pending = Vec::new();
        for out in &outs {
            pending.extend_from_slice(&out.pending);
        }
        mint_reactions(self, &pending);
        for out in outs {
            for mut child in out.births {
                if self.agents.len() >= self.cfg.max_agents {
                    // Capacity reached: the child's body mass returns to its cell as soil.
                    let cell = self.grid.idx(child.x, child.y);
                    self.grid.cells[cell].add(crate::chem::generate::MAT_SOIL, child.body);
                    continue;
                }
                child.id = self.next_id;
                self.next_id += 1;
                self.agents.push(child);
            }
        }
        // 5. fields
        diffuse_soil(&mut self.grid);
        diffuse_water(&mut self.grid);
        relax_temperature(&mut self.grid, TEMP_DECAY);
        // 6. external events
        self.apply_events();
        // 7.
        self.tick += 1;
    }

    pub fn run(&mut self, ticks: u64) {
        for _ in 0..ticks {
            self.step();
        }
    }

    pub(crate) fn apply_events(&mut self) {
        let tick = self.tick;
        let evs: Vec<crate::events::ExternalEvent> = self.events.take_for(tick).iter().map(|e| e.1.clone()).collect();
        for e in &evs {
            crate::events::apply(self, e);
        }
    }
}
