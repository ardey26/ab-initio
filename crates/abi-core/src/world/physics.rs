//! Cell physics that runs with zero agents: erosion, weathering reactions, fire,
//! autotroph growth. Runs per chunk in parallel; reactions that mint materials
//! are collected and minted sequentially in chunk order.

use super::cell::Cell;
use crate::chem::generate::{MAT_AUTOTROPH, MAT_SOIL};
use crate::chem::props::*;
use crate::chem::{Chemistry, HOT_TEMP};
use crate::rng::hash3;
use crate::stats::Stats;
use crate::world::World;

pub const EROSION: u32 = 10;
pub const WEATHER_RATE: u32 = 10;
pub const COLD_WEATHER_PERIOD: u64 = 32;
pub const AUTOTROPH_CAP: u32 = 3000;
pub const FIRE_ENERGY: f32 = 0.6;
pub const BURN_RATE: u32 = 20;
pub const FIRE_HEAT: f32 = 0.1;
pub const TEMP_DECAY: f32 = 0.85;

#[derive(Clone, Copy, Debug)]
pub struct PendingReaction {
    pub cell: usize,
    pub a: MatId,
    pub b: MatId,
    pub temp: f32,
    pub hot: bool,
}

pub fn chunk_physics(cells: &mut [Cell], cell_base: usize, seed: u64, tick: u64, chem: &Chemistry, growth: u32, stats: &mut Stats, pending: &mut Vec<PendingReaction>) {
    for (li, c) in cells.iter_mut().enumerate() {
        let ci = cell_base + li;
        // Erosion.
        if c.bedrock > 0 {
            let e = c.bedrock.min(EROSION);
            c.bedrock -= e;
            let ore = c.ore;
            c.add(ore, e);
        }
        // Fire: first loose non-soil item that is energetic and above both its
        // melting point and HOT_TEMP. Ambient cells never ignite; a heat source
        // (volcanic cell, heat action, an adjacent fire via FIRE_HEAT) is needed.
        let fuel = c.inv.iter().filter(|e| e.0 != MAT_SOIL && e.1 > 0).find(|e| {
            let p = chem.props(e.0);
            p[P_ENERGY] > FIRE_ENERGY && c.temp > p[P_MELT].max(HOT_TEMP)
        }).map(|e| e.0);
        if let Some(f) = fuel {
            let burned = c.remove(f, BURN_RATE);
            c.add(MAT_SOIL, burned);
            c.temp = (c.temp + FIRE_HEAT).min(1.5);
            stats.fires += 1;
        }
        // Weathering.
        let top = c.top2_non_soil();
        let hot = Chemistry::hot(c.temp);
        let cold_roll = hash3(seed ^ 0x3EA7_0000, tick, ci as u64) % COLD_WEATHER_PERIOD == 0;
        if top[0].1 >= WEATHER_RATE && top[1].1 >= WEATHER_RATE && (hot || cold_roll) {
            c.remove(top[0].0, WEATHER_RATE);
            c.remove(top[1].0, WEATHER_RATE);
            pending.push(PendingReaction { cell: ci, a: top[0].0, b: top[1].0, temp: c.temp, hot });
        }
        // Growth.
        if c.fertile() && growth > 0 {
            let cur = c.get(MAT_AUTOTROPH);
            if cur < AUTOTROPH_CAP {
                let g = c.remove(MAT_SOIL, growth.min(AUTOTROPH_CAP - cur));
                c.add(MAT_AUTOTROPH, g);
            }
        }
    }
}

/// Sequential: mint products in the given (chunk) order.
pub fn mint_reactions(w: &mut World, pending: &[PendingReaction]) {
    for p in pending {
        let id = w.chem.combine(p.a, p.b, p.temp);
        w.grid.cells[p.cell].add(id, 2 * WEATHER_RATE);
        w.stats.env_reactions += 1;
        if p.hot {
            w.stats.env_hot += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chem::generate::{ChemParams, MAT_AUTOTROPH, MAT_SOIL};
    use crate::stats::Stats;
    use crate::world::config::WorldConfig;
    use crate::world::World;

    fn world() -> World {
        World::new(&WorldConfig { seed: 3, width: 32, height: 32, pop0: 0, chem: ChemParams { n_base: 12, ..Default::default() }, ..Default::default() })
    }

    fn run(w: &mut World) -> Vec<PendingReaction> {
        let mut pending = Vec::new();
        let mut stats = Stats::default();
        let (seed, tick, growth) = (w.cfg.seed, w.tick, w.growth);
        chunk_physics(&mut w.grid.cells[..], 0, seed, tick, &w.chem, growth, &mut stats, &mut pending);
        w.stats.merge(&stats);
        pending
    }

    #[test]
    fn erosion_moves_mass_from_bedrock_to_ore_exactly() {
        let mut w = world();
        let c = w.grid.cells.iter().position(|c| c.bedrock > 0).unwrap();
        let (b0, ore) = (w.grid.cells[c].bedrock, w.grid.cells[c].ore);
        let o0 = w.grid.cells[c].get(ore);
        let m0 = w.total_mass();
        let pending = run(&mut w);
        assert_eq!(w.grid.cells[c].bedrock, b0 - EROSION);
        assert!(w.grid.cells[c].get(ore) >= o0 + EROSION - WEATHER_RATE);
        mint_reactions(&mut w, &pending);
        assert_eq!(w.total_mass(), m0);
    }

    #[test]
    fn growth_only_in_fertile_cells_and_capped() {
        let mut w = world();
        let fert = w.grid.cells.iter().position(|c| c.fertile()).unwrap();
        let dry = w.grid.cells.iter().position(|c| !c.fertile()).unwrap();
        w.grid.cells[fert].inv.clear();
        w.grid.cells[fert].add(MAT_SOIL, 5000);
        w.grid.cells[dry].inv.clear();
        w.grid.cells[dry].add(MAT_SOIL, 5000);
        let g = w.growth;
        let _ = run(&mut w);
        assert_eq!(w.grid.cells[fert].get(MAT_AUTOTROPH), g);
        assert_eq!(w.grid.cells[dry].get(MAT_AUTOTROPH), 0);
        w.grid.cells[fert].add(MAT_AUTOTROPH, AUTOTROPH_CAP);
        let before = w.grid.cells[fert].get(MAT_AUTOTROPH);
        let _ = run(&mut w);
        assert!(w.grid.cells[fert].get(MAT_AUTOTROPH) <= before, "no growth above cap");
    }

    #[test]
    fn hot_cells_react_every_tick_and_products_conserve_mass() {
        let mut w = world();
        let c = 7;
        w.grid.cells[c].inv.clear();
        w.grid.cells[c].add(MAT_SOIL, 1000);
        w.grid.cells[c].add(2, 500);
        w.grid.cells[c].add(5, 500);
        w.grid.cells[c].temp = 0.9;
        w.grid.cells[c].bedrock = 0;
        let m0 = w.total_mass();
        let pending = run(&mut w);
        let mine: Vec<_> = pending.iter().filter(|p| p.cell == c).collect();
        assert_eq!(mine.len(), 1);
        assert!(mine[0].hot);
        assert_eq!(w.grid.cells[c].get(2), 500 - WEATHER_RATE);
        mint_reactions(&mut w, &pending);
        assert_eq!(w.total_mass(), m0);
        assert_eq!(w.stats.env_reactions, pending.len() as u64);
        let product = w.chem.combine(2, 5, 0.9);
        assert_eq!(w.grid.cells[c].get(product), 2 * WEATHER_RATE);
    }

    #[test]
    fn soil_only_cell_never_reacts() {
        let mut w = world();
        for c in w.grid.cells.iter_mut() {
            c.inv.clear();
            c.add(MAT_SOIL, 1000);
            c.temp = 0.9;
            c.bedrock = 0;
            c.water = 0;
        }
        let pending = run(&mut w);
        assert!(pending.is_empty());
    }

    #[test]
    fn fire_burns_energetic_material_above_its_melting_point() {
        let mut w = world();
        let c = 9;
        let fuel_raw = { let mut p = [0f32; NP]; p[P_ENERGY] = 3.0; p[P_MELT] = -2.0; p };
        let fuel = w.chem.table.intern(fuel_raw, crate::chem::table::Recipe { a: 1, b: 3, tq: 0 });
        for (i, other) in w.grid.cells.iter_mut().enumerate() {
            if i != c {
                other.inv.clear();
                other.bedrock = 0;
            }
        }
        w.grid.cells[c].inv.clear();
        w.grid.cells[c].add(fuel, 100);
        w.grid.cells[c].temp = 0.5;
        w.grid.cells[c].bedrock = 0;
        w.grid.cells[c].water = 0;
        let m0 = w.total_mass();
        let _ = run(&mut w);
        assert_eq!(w.grid.cells[c].get(fuel), 100 - BURN_RATE);
        assert_eq!(w.grid.cells[c].get(MAT_SOIL), BURN_RATE);
        assert!(w.grid.cells[c].temp > 0.5);
        assert_eq!(w.stats.fires, 1);
        assert_eq!(w.total_mass(), m0);
    }

    #[test]
    fn fire_needs_a_heat_source_above_ambient_to_ignite() {
        let mut w = world();
        let c = 9;
        let fuel_raw = { let mut p = [0f32; NP]; p[P_ENERGY] = 3.0; p[P_MELT] = -8.0; p };
        let fuel = w.chem.table.intern(fuel_raw, crate::chem::table::Recipe { a: 1, b: 3, tq: 0 });
        assert!(w.chem.props(fuel)[P_MELT] < crate::world::generate::AMBIENT, "fuel melts below ambient");
        for other in w.grid.cells.iter_mut() {
            other.inv.clear();
            other.bedrock = 0;
        }
        w.grid.cells[c].add(fuel, 100);
        w.grid.cells[c].temp = crate::world::generate::AMBIENT;
        let _ = run(&mut w);
        assert_eq!(w.grid.cells[c].get(fuel), 100, "no fire at ambient temperature");
        assert_eq!(w.stats.fires, 0);
    }
}
