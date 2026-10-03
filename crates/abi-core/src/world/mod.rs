pub mod cell;
pub mod config;
pub mod fields;
pub mod generate;
pub mod grid;

use crate::agent::genome::Genome;
use crate::agent::memory::SocialMemory;
use crate::agent::{Agent, BODY_TARGET, NO_ACTION, START_ENERGY};
use crate::chem::generate::MAT_SOIL;
use crate::chem::Chemistry;
use crate::rng::{hash3, Rng};
use crate::stats::Stats;
use config::WorldConfig;
use grid::{Grid, CHUNK};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct World {
    pub cfg: WorldConfig,
    pub tick: u64,
    pub grid: Grid,
    pub chem: Chemistry,
    pub agents: Vec<Agent>,
    pub next_id: u64,
    pub growth: u32,
    #[serde(skip)]
    pub stats: Stats,
    pub agent_start: Vec<u32>,
    pub cell_start: Vec<u32>,
}

impl World {
    pub fn new(cfg: &WorldConfig) -> Self {
        let terrain = generate::generate_terrain(cfg);
        let chem = Chemistry::new(cfg.seed, &cfg.chem);
        let mut grid = terrain.grid;
        let mut agents = Vec::with_capacity(cfg.pop0);
        for i in 0..cfg.pop0 {
            let mut r = Rng::new(hash3(cfg.seed ^ 0xA6E7_0000, i as u64, 0));
            let x = r.range(cfg.width) as u16;
            let y = r.range(cfg.height) as u16;
            let genome = Genome::random(&mut r);
            let cell = grid.idx(x, y);
            let body = grid.cells[cell].remove(MAT_SOIL, BODY_TARGET);
            agents.push(Agent { alive: true, id: i as u64, parent: u64::MAX, x, y, energy: START_ENERGY, age: 0, body, held: [(0, 0); 2], signal: [0.0; 2], last_action: NO_ACTION, genome, memory: SocialMemory::new() });
        }
        let n_chunks = grid.n_chunks();
        let n_cells = grid.len();
        let mut w = World { cfg: cfg.clone(), tick: 0, grid, chem, agents, next_id: cfg.pop0 as u64, growth: terrain.growth_per_tick, stats: Stats::default(), agent_start: vec![0; n_chunks + 1], cell_start: vec![0; n_cells + 1] };
        w.sort_agents();
        w
    }

    /// Physically sort living agents by (cell, id); drop dead ones; rebuild ranges.
    pub fn sort_agents(&mut self) {
        let grid = &self.grid;
        let mut keyed: Vec<(usize, u64, u32)> = self.agents.iter().enumerate().filter(|(_, a)| a.alive).map(|(i, a)| (grid.idx(a.x, a.y), a.id, i as u32)).collect();
        keyed.sort_unstable();
        let old = std::mem::take(&mut self.agents);
        let mut old: Vec<Option<Agent>> = old.into_iter().map(Some).collect();
        self.agents = keyed.iter().map(|&(_, _, i)| old[i as usize].take().unwrap()).collect();
        self.cell_start.fill(0);
        self.agent_start.fill(0);
        for &(cell, _, _) in &keyed {
            self.cell_start[cell + 1] += 1;
            self.agent_start[self.grid.chunk_of(cell) + 1] += 1;
        }
        for c in 0..self.grid.len() {
            self.cell_start[c + 1] += self.cell_start[c];
        }
        for ch in 0..self.grid.n_chunks() {
            self.agent_start[ch + 1] += self.agent_start[ch];
        }
    }

    pub fn agents_in(&self, cell: usize) -> &[Agent] {
        &self.agents[self.cell_start[cell] as usize..self.cell_start[cell + 1] as usize]
    }

    pub fn population(&self) -> usize {
        self.agents.iter().filter(|a| a.alive).count()
    }

    /// Cells (loose + bedrock + water) + agents (body + held).
    pub fn total_mass(&self) -> u64 {
        let cells: u64 = self.grid.cells.iter().map(|c| c.mass()).sum();
        let agents: u64 = self.agents.iter().filter(|a| a.alive).map(|a| a.body as u64 + a.held[0].1 as u64 + a.held[1].1 as u64).sum();
        cells + agents
    }

    pub const CHUNK_CELLS: usize = CHUNK * CHUNK;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::config::WorldConfig;

    #[test]
    fn new_world_takes_agent_bodies_from_soil_and_sorts() {
        let cfg = WorldConfig { seed: 2, width: 32, height: 32, pop0: 100, ..Default::default() };
        let w = World::new(&cfg);
        assert_eq!(w.population(), 100);
        let soil_total: u64 = w.grid.cells.iter().map(|c| c.get(crate::chem::generate::MAT_SOIL) as u64).sum();
        let bodies: u64 = w.agents.iter().map(|a| a.body as u64).sum();
        assert_eq!(soil_total + bodies, 1024 * crate::world::generate::SOIL0 as u64);
        for k in 1..w.agents.len() {
            let (a, b) = (&w.agents[k - 1], &w.agents[k]);
            let (ca, cb) = (w.grid.idx(a.x, a.y), w.grid.idx(b.x, b.y));
            assert!((ca, a.id) < (cb, b.id), "agents sorted by (cell, id)");
        }
        assert_eq!(w.agent_start.len(), w.grid.n_chunks() + 1);
        assert_eq!(*w.agent_start.last().unwrap() as usize, 100);
        assert_eq!(*w.cell_start.last().unwrap() as usize, 100);
    }
}
