//! Digestion, excretion, death and asexual reproduction. Runs per chunk in
//! parallel; children are collected and placed sequentially in Task 14.

use super::actions::ChunkCtx;
use super::memory::SocialMemory;
use super::*;
use crate::chem::generate::MAT_SOIL;
use crate::chem::props::*;
use crate::rng::{hash3, Rng};

const EXCRETE_MARGIN: u32 = 200;

pub fn metabolize_chunk(ctx: &mut ChunkCtx, seed: u64, births: &mut Vec<Agent>) {
    let n = ctx.agents.len();
    for la in 0..n {
        if !ctx.agents[la].alive {
            continue;
        }
        let (x, y, id) = {
            let a = &ctx.agents[la];
            (a.x, a.y, a.id)
        };
        let here = ctx.grid.idx(x, y);
        let n_weights = ctx.agents[la].genome.n_weights() as f32;
        let a = &mut ctx.agents[la];
        a.energy -= BASE_COST + COGNITION_COST_PER_WEIGHT * n_weights;
        a.age += 1;
        for s in 0..2 {
            let (mid, m) = a.held[s];
            if m == 0 {
                continue;
            }
            let p = ctx.chem.props(mid);
            if p[P_NUTRI] > a.genome.digest_thr {
                let d = m.min(DIGEST_RATE);
                let gain = (p[P_NUTRI] - 1.5 * p[P_TOXIC]) * d as f32 / 1000.0 * ENERGY_PER_UNIT;
                a.energy += gain;
                a.body += d;
                a.held[s].1 -= d;
                if a.held[s].1 == 0 {
                    a.held[s] = (0, 0);
                }
                let g = gain.max(0.0) as f64;
                ctx.stats.energy_total += g;
                ctx.stats.eaten_by_material.push((mid, g));
                ctx.stats.materials_in_use.push(mid);
                if ctx.chem.table.is_artifact(mid) {
                    ctx.stats.energy_from_artifacts += g;
                }
            }
        }
        let target = a.body_target();
        let cap = target + BODY_REPRO_EXTRA + EXCRETE_MARGIN;
        if a.body > cap {
            let ex = a.body - cap;
            a.body -= ex;
            ctx.cells[here - ctx.cell_base].add(MAT_SOIL, ex);
        }
        let a = &mut ctx.agents[la];
        if a.energy <= 0.0 || a.age > LIFESPAN {
            a.alive = false;
            let body = a.body;
            let held = a.held;
            a.body = 0;
            a.held = [(0, 0); 2];
            let cell = &mut ctx.cells[here - ctx.cell_base];
            cell.add(MAT_SOIL, body);
            for (hid, hm) in held {
                cell.add(hid, hm);
            }
            ctx.stats.deaths += 1;
            continue;
        }
        if a.energy >= REPRO_ENERGY && a.body >= target + BODY_REPRO_EXTRA {
            a.energy *= 0.5;
            let child_body = target;
            a.body -= child_body;
            let mut rng = Rng::new(hash3(seed ^ 0xB1B7_0000, ctx.tick, id));
            let genome = a.genome.mutate(&mut rng);
            let d = rng.range(4) as u8;
            let (cx, cy) = ctx.grid.step(x, y, d);
            births.push(Agent { alive: true, id: u64::MAX, parent: id, x: cx, y: cy, energy: a.energy, age: 0, body: child_body, held: [(0, 0); 2], signal: [0.0; 2], last_action: NO_ACTION, recent: 0, genome, memory: SocialMemory::new() });
            ctx.stats.births += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::actions::ChunkCtx;
    use crate::chem::generate::{ChemParams, MAT_AUTOTROPH, MAT_SOIL};
    use crate::stats::Stats;
    use crate::world::config::WorldConfig;
    use crate::world::grid::Grid;
    use crate::world::World;

    fn world(pop: usize) -> World {
        World::new(&WorldConfig { seed: 8, width: 32, height: 32, pop0: pop, chem: ChemParams { n_base: 12, ..Default::default() }, ..Default::default() })
    }

    fn run_chunk0(w: &mut World) -> (Vec<Agent>, Stats) {
        let mut births = Vec::new();
        let mut stats = Stats::default();
        let mut deferred = Vec::new();
        let (a0, a1) = (w.agent_start[0] as usize, w.agent_start[1] as usize);
        let seed = w.cfg.seed;
        let World { grid, agents, chem, cell_start, tick, .. } = w;
        let grid_ro = Grid { width: grid.width, height: grid.height, cells: Vec::new() };
        let cells = &mut grid.cells[0..World::CHUNK_CELLS];
        let mut ctx = ChunkCtx { cells, cell_base: 0, agents: &mut agents[a0..a1], agent_base_cell_start: &cell_start[..], chem, grid: &grid_ro, tick: *tick, stats: &mut stats, deferred: &mut deferred };
        metabolize_chunk(&mut ctx, seed, &mut births);
        (births, stats)
    }

    #[test]
    fn digestion_moves_mass_to_body_and_raises_energy() {
        let mut w = world(1);
        w.agents[0].genome.digest_thr = 0.1;
        w.agents[0].held[0] = (MAT_AUTOTROPH, 250);
        let (e0, b0, m0) = (w.agents[0].energy, w.agents[0].body, w.total_mass());
        let (_, stats) = run_chunk0(&mut w);
        assert_eq!(w.agents[0].held[0], (MAT_AUTOTROPH, 150));
        assert_eq!(w.agents[0].body, b0 + DIGEST_RATE);
        assert!(w.agents[0].energy > e0, "net gain while eating autotroph");
        assert_eq!(w.total_mass(), m0);
        assert!(stats.energy_total > 0.0 && stats.energy_from_artifacts == 0.0);
    }

    #[test]
    fn death_returns_all_mass_to_the_cell() {
        let mut w = world(1);
        w.agents[0].energy = 0.5; // dies this tick after base cost
        w.agents[0].genome.digest_thr = f32::MAX; // digest nothing, so held mass is returned intact
        w.agents[0].held = [(MAT_AUTOTROPH, 120), (3, 40)];
        let cell = w.grid.idx(w.agents[0].x, w.agents[0].y);
        let soil_before = w.grid.cells[cell].get(MAT_SOIL);
        let body = w.agents[0].body;
        let m0 = w.total_mass();
        let (_, stats) = run_chunk0(&mut w);
        assert!(!w.agents[0].alive);
        assert_eq!(stats.deaths, 1);
        assert_eq!(w.grid.cells[cell].get(MAT_SOIL), soil_before + body);
        assert_eq!(w.grid.cells[cell].get(MAT_AUTOTROPH) >= 120, true);
        assert_eq!(w.grid.cells[cell].get(3) >= 40, true);
        assert_eq!(w.total_mass(), m0);
    }

    #[test]
    fn reproduction_requires_energy_and_body_and_splits_both() {
        let mut w = world(1);
        let a = &mut w.agents[0];
        a.energy = REPRO_ENERGY + 10.0;
        a.body = a.body_target() + BODY_REPRO_EXTRA;
        let body_before = a.body;
        let m0 = w.total_mass();
        let (births, stats) = run_chunk0(&mut w);
        assert_eq!(births.len(), 1);
        assert_eq!(stats.births, 1);
        let child = &births[0];
        assert_eq!(child.parent, 0);
        assert_eq!(child.id, u64::MAX, "id assigned later in chunk order");
        assert_eq!(child.body + w.agents[0].body, body_before);
        assert!((child.energy - w.agents[0].energy).abs() < 1e-3);
        // Mass conservation must count births too.
        assert_eq!(w.total_mass() + child.body as u64, m0);
    }

    #[test]
    fn cognition_costs_scale_with_weights() {
        let mut w = world(2);
        w.agents[0].held = [(0, 0); 2];
        w.agents[1].held = [(0, 0); 2];
        // Grow agent 1's hidden layer a lot.
        for _ in 0..10 {
            let mut g = w.agents[1].genome.clone();
            g.hidden = (g.hidden + 1).min(crate::agent::brain::NH_MAX);
            let h = g.hidden as usize;
            g.w1 = vec![0; h * crate::agent::brain::NIN];
            g.b1 = vec![0; h];
            g.w2 = vec![0; crate::agent::brain::NOUT * h];
            w.agents[1].genome = g;
        }
        let e = [w.agents[0].energy, w.agents[1].energy];
        let ids = [w.agents[0].id, w.agents[1].id];
        run_chunk0(&mut w);
        // Both agents are in chunk 0 (32x32 world has one chunk).
        let get = |id: u64| w.agents.iter().find(|a| a.id == id).unwrap().energy;
        let cost0 = e[0] - get(ids[0]);
        let cost1 = e[1] - get(ids[1]);
        assert!(cost1 > cost0, "bigger brain must cost more: {} vs {}", cost1, cost0);
    }
}
