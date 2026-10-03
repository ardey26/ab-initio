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
                let c = w.grid.idx(*x, *y);
                w.grid.cells[c].add(*material, *mass);
                w.external_mass += *mass as i64;
            }
        }
        ExternalEvent::Impact { cx, cy, radius } => {
            let cells = disc(w, *cx, *cy, *radius);
            for &c in &cells {
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
            let rim: Vec<usize> = disc(w, *cx, *cy, *radius).into_iter().filter(|c| !disc(w, *cx, *cy, radius.saturating_sub(1)).contains(c)).collect();
            for c in rim {
                let e = &mut w.grid.cells[c].elevation;
                *e = (*e + 0.1).min(1.0);
            }
        }
        ExternalEvent::SeedOrganism { x, y } => {
            let c = w.grid.idx(*x, *y);
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
            let a = Agent { alive: true, id: w.next_id, parent: u64::MAX, x: *x, y: *y, energy: START_ENERGY, age: 0, body, held: [(0, 0); 2], signal: [0.0; 2], last_action: NO_ACTION, recent: 0, genome, memory: SocialMemory::new() };
            w.next_id += 1;
            w.agents.push(a);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
