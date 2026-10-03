//! Interventions. All of them act through physics and are logged so a replay with
//! the same log reproduces the same history. Matter they add is tracked in
//! `World::external_mass` so conservation stays checkable.

use crate::agent::genome::Genome;
use crate::agent::memory::SocialMemory;
use crate::agent::{Agent, BODY_TARGET, NO_ACTION, START_ENERGY};
use crate::chem::generate::{MAT_CRUST, MAT_SOIL};
use crate::chem::props::MatId;
use crate::rng::{hash3, Rng};
use crate::world::World;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExternalEvent {
    Rain { cx: u16, cy: u16, radius: u16, water_per_cell: u32 },
    Temperature { cx: u16, cy: u16, radius: u16, delta: f32 },
    DropMatter { x: u16, y: u16, material: MatId, mass: u32 },
    Impact { cx: u16, cy: u16, radius: u16 },
    SeedOrganism { x: u16, y: u16 },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EventLog {
    pub entries: Vec<(u64, ExternalEvent)>,
    pub cursor: usize,
}

impl EventLog {
    pub fn push(&mut self, tick: u64, ev: ExternalEvent) {
        if let Some(last) = self.entries.last() {
            assert!(tick >= last.0, "events must be pushed in tick order");
        }
        self.entries.push((tick, ev));
    }

    pub fn take_for(&mut self, tick: u64) -> &[(u64, ExternalEvent)] {
        let start = self.cursor;
        while self.cursor < self.entries.len() && self.entries[self.cursor].0 <= tick {
            self.cursor += 1;
        }
        &self.entries[start..self.cursor]
    }
}

/// Cells within Chebyshev distance `radius` of (cx, cy) on the torus, in index order.
fn disc(w: &World, cx: u16, cy: u16, radius: u16) -> Vec<usize> {
    let mut v = Vec::new();
    let r = radius as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            let x = ((cx as i32 + dx).rem_euclid(w.grid.width as i32)) as u16;
            let y = ((cy as i32 + dy).rem_euclid(w.grid.height as i32)) as u16;
            v.push(w.grid.idx(x, y));
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Cell index for coordinates wrapped onto the torus, like the disc-based events.
fn wrapped_idx(w: &World, x: u16, y: u16) -> usize {
    let x = (x as usize % w.grid.width) as u16;
    let y = (y as usize % w.grid.height) as u16;
    w.grid.idx(x, y)
}

pub fn apply(w: &mut World, ev: &ExternalEvent) {
    match ev {
        ExternalEvent::Rain { cx, cy, radius, water_per_cell } => {
            for c in disc(w, *cx, *cy, *radius) {
                w.grid.cells[c].water += water_per_cell;
                w.external_mass += *water_per_cell as i64;
            }
        }
        ExternalEvent::Temperature { cx, cy, radius, delta } => {
            for c in disc(w, *cx, *cy, *radius) {
                let t = &mut w.grid.cells[c].temp;
                *t = (*t + delta).clamp(0.0, 1.5);
            }
        }
        ExternalEvent::DropMatter { x, y, material, mass } => {
            if (*material as usize) < w.chem.table.len() && !w.chem.table.is_evicted(*material) {
                let c = wrapped_idx(w, *x, *y);
                w.grid.cells[c].add(*material, *mass);
                w.external_mass += *mass as i64;
            }
        }
        ExternalEvent::Impact { cx, cy, radius } => {
            let outer = disc(w, *cx, *cy, *radius);
            let inner = disc(w, *cx, *cy, radius.saturating_sub(1));
            for &c in &outer {
                let cell = &mut w.grid.cells[c];
                let loose: Vec<(MatId, u32)> = cell.inv.iter().copied().filter(|e| e.0 != MAT_SOIL).collect();
                let target = if cell.bedrock > 0 { cell.ore } else { MAT_CRUST };
                for (id, m) in loose {
                    cell.remove(id, m);
                    cell.add(target, m / 2);
                    cell.add(MAT_SOIL, m - m / 2);
                }
                cell.temp = 1.5;
            }
            // The rim is the outer ring; a radius-0 impact raises its single cell.
            for &c in &outer {
                if *radius == 0 || inner.binary_search(&c).is_err() {
                    let e = &mut w.grid.cells[c].elevation;
                    *e = (*e + 0.1).min(1.0);
                }
            }
        }
        ExternalEvent::SeedOrganism { x, y } => {
            let c = wrapped_idx(w, *x, *y);
            let (x, y) = w.grid.xy(c);
            let body = w.grid.cells[c].remove(MAT_SOIL, BODY_TARGET);
            if body == 0 {
                return;
            }
            let mut r = Rng::new(hash3(w.cfg.seed ^ 0x5EED_0000, w.tick, c as u64));
            let mut genome = Genome::random(&mut r);
            genome.hidden = crate::agent::brain::NH_MIN;
            let h = genome.hidden as usize;
            genome.w1.truncate(h * crate::agent::brain::NIN);
            genome.b1.truncate(h);
            genome.w2 = (0..crate::agent::brain::NOUT * h).map(|_| (r.normal() * 0.3 / genome.scale).round().clamp(-127.0, 127.0) as i8).collect();
            let a = Agent { alive: true, id: w.next_id, parent: u64::MAX, x, y, energy: START_ENERGY, age: 0, body, held: [(0, 0); 2], signal: [0.0; 2], last_action: NO_ACTION, recent: 0, genome, memory: SocialMemory::new() };
            w.next_id += 1;
            w.agents.push(a);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::agent::BODY_TARGET;
    use crate::chem::generate::MAT_SOIL;
    use crate::world::config::WorldConfig;

    fn empty_world() -> World {
        World::new(&WorldConfig { seed: 9, width: 32, height: 32, pop0: 0, ..Default::default() })
    }

    #[test]
    fn impact_converts_loose_matter_and_heats() {
        let mut w = empty_world();
        let (cx, cy) = (10u16, 10u16);
        let ore: MatId = 7;
        let centre = w.grid.idx(cx, cy);
        let rim = w.grid.idx(cx + 1, cy);
        let outside = w.grid.idx(cx + 2, cy);
        for c in [centre, rim, outside] {
            w.grid.cells[c].elevation = 0.5;
        }
        {
            let cell = &mut w.grid.cells[centre];
            cell.inv.retain(|e| e.0 == MAT_SOIL);
            cell.bedrock = 1000;
            cell.ore = ore;
            cell.add(5, 400);
            cell.add(6, 200);
        }
        let soil0 = w.grid.cells[centre].get(MAT_SOIL);
        let ore0 = w.grid.cells[centre].get(ore);
        let total0 = w.total_mass();
        apply(&mut w, &ExternalEvent::Impact { cx, cy, radius: 1 });
        let cell = &w.grid.cells[centre];
        assert_eq!(cell.get(5), 0);
        assert_eq!(cell.get(6), 0);
        assert_eq!(cell.get(ore), ore0 + 400 / 2 + 200 / 2);
        assert_eq!(cell.get(MAT_SOIL), soil0 + 200 + 100);
        assert_eq!(cell.temp, 1.5);
        assert_eq!(w.total_mass(), total0);
        assert_eq!(w.external_mass, 0);
        assert!((w.grid.cells[rim].elevation - 0.6).abs() < 1e-6, "rim rises");
        assert_eq!(w.grid.cells[centre].elevation, 0.5, "interior does not");
        assert_eq!(w.grid.cells[outside].elevation, 0.5, "outside the disc does not");
    }

    #[test]
    fn impact_without_bedrock_converts_to_crust_and_radius_zero_raises_centre() {
        let mut w = empty_world();
        let c = w.grid.idx(4, 4);
        w.grid.cells[c].inv.retain(|e| e.0 == MAT_SOIL);
        w.grid.cells[c].bedrock = 0;
        w.grid.cells[c].elevation = 0.95;
        w.grid.cells[c].add(5, 100);
        let crust0 = w.grid.cells[c].get(MAT_CRUST);
        apply(&mut w, &ExternalEvent::Impact { cx: 4, cy: 4, radius: 0 });
        assert_eq!(w.grid.cells[c].get(MAT_CRUST), crust0 + 50);
        assert_eq!(w.grid.cells[c].elevation, 1.0, "clamped to 1");
    }

    #[test]
    fn seed_organism_takes_soil_and_adds_agent() {
        let mut w = empty_world();
        let c = w.grid.idx(3, 3);
        w.grid.cells[c].add(MAT_SOIL, BODY_TARGET);
        let soil0 = w.grid.cells[c].get(MAT_SOIL);
        let m0 = w.conserved_mass();
        apply(&mut w, &ExternalEvent::SeedOrganism { x: 3, y: 3 });
        assert_eq!(w.population(), 1);
        assert_eq!(w.agents[0].id, w.cfg.pop0 as u64);
        assert_eq!(w.grid.cells[c].get(MAT_SOIL), soil0 - BODY_TARGET);
        assert_eq!(w.conserved_mass(), m0);
        let d = w.grid.idx(5, 5);
        w.grid.cells[d].inv.clear();
        apply(&mut w, &ExternalEvent::SeedOrganism { x: 5, y: 5 });
        assert_eq!(w.population(), 1, "no soil, no organism");
    }

    #[test]
    fn drop_matter_ignores_invalid_material() {
        let mut w = empty_world();
        let c = w.grid.idx(2, 2);
        let total0 = w.total_mass();
        let invalid = w.chem.table.len() as MatId;
        apply(&mut w, &ExternalEvent::DropMatter { x: 2, y: 2, material: invalid, mass: 3000 });
        assert_eq!(w.external_mass, 0);
        assert_eq!(w.total_mass(), total0);
        let before = w.grid.cells[c].get(5);
        apply(&mut w, &ExternalEvent::DropMatter { x: 2, y: 2, material: 5, mass: 3000 });
        assert_eq!(w.grid.cells[c].get(5), before + 3000);
        assert_eq!(w.external_mass, 3000);
    }

    #[test]
    fn rain_ledger_matches_distinct_cells_when_disc_wraps() {
        let mut w = empty_world();
        apply(&mut w, &ExternalEvent::Rain { cx: 0, cy: 0, radius: 2, water_per_cell: 10 });
        assert_eq!(w.external_mass, 25 * 10);
    }

    #[test]
    fn out_of_range_coordinates_wrap() {
        let mut w = empty_world();
        let c = w.grid.idx(1, 1);
        let before = w.grid.cells[c].get(5);
        apply(&mut w, &ExternalEvent::DropMatter { x: 33, y: 33, material: 5, mass: 10 });
        assert_eq!(w.grid.cells[c].get(5), before + 10);
    }

    #[test]
    fn log_is_ordered_and_cursor_advances() {
        let mut log = EventLog::default();
        log.push(5, ExternalEvent::Temperature { cx: 1, cy: 1, radius: 2, delta: 0.5 });
        log.push(5, ExternalEvent::SeedOrganism { x: 0, y: 0 });
        log.push(9, ExternalEvent::Rain { cx: 0, cy: 0, radius: 1, water_per_cell: 100 });
        assert!(log.take_for(4).is_empty());
        assert_eq!(log.take_for(5).len(), 2);
        assert!(log.take_for(6).is_empty());
        assert_eq!(log.take_for(9).len(), 1);
    }

    #[test]
    #[should_panic]
    fn log_rejects_out_of_order_push() {
        let mut log = EventLog::default();
        log.push(5, ExternalEvent::SeedOrganism { x: 0, y: 0 });
        log.push(4, ExternalEvent::SeedOrganism { x: 0, y: 0 });
    }
}
