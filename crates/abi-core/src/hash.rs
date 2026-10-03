use crate::events::ExternalEvent;
use crate::rng::mix64;
use crate::world::World;

pub fn state_hash(w: &World) -> u64 {
    let mut h = 0x1234_5678_9abc_def0u64 ^ w.tick;
    let mut mix = |v: u64| h = mix64(h ^ v.wrapping_add(0x9e37_79b9_7f4a_7c15));
    for a in &w.agents {
        if !a.alive {
            continue;
        }
        mix(a.id);
        mix(a.x as u64 | (a.y as u64) << 16);
        mix(a.energy.to_bits() as u64);
        mix(a.body as u64);
        mix(a.held[0].0 as u64 | (a.held[0].1 as u64) << 32);
        mix(a.held[1].0 as u64 | (a.held[1].1 as u64) << 32);
        mix(a.genome.hash());
        mix(a.age as u64);
        mix(a.parent);
        mix(a.signal[0].to_bits() as u64 | (a.signal[1].to_bits() as u64) << 32);
        mix(a.last_action as u64);
        for s in &a.memory.slots {
            mix(s.0);
            for v in s.1 {
                mix(v.to_bits() as u64);
            }
            mix(s.2 as u64);
        }
    }
    for c in &w.grid.cells {
        // Inventory order is state: swap_remove and fire both depend on it.
        for &(id, m) in &c.inv {
            mix(id as u64 | (m as u64) << 32);
        }
        mix(c.inv.len() as u64);
        mix(c.water as u64 | (c.bedrock as u64) << 32);
        mix(c.temp.to_bits() as u64);
        mix(c.ambient.to_bits() as u64);
        mix(c.elevation.to_bits() as u64);
        mix(c.ore as u64);
    }
    // Pending interventions decide future ticks, and the ledger is part of the invariant.
    mix(w.events.cursor as u64);
    mix(w.events.entries.len() as u64);
    for (tick, ev) in &w.events.entries[w.events.cursor..] {
        mix(*tick);
        match ev {
            ExternalEvent::Rain { cx, cy, radius, water_per_cell } => {
                mix(0);
                mix(*cx as u64 | (*cy as u64) << 16 | (*radius as u64) << 32);
                mix(*water_per_cell as u64);
            }
            ExternalEvent::Temperature { cx, cy, radius, delta } => {
                mix(1);
                mix(*cx as u64 | (*cy as u64) << 16 | (*radius as u64) << 32);
                mix(delta.to_bits() as u64);
            }
            ExternalEvent::DropMatter { x, y, material, mass } => {
                mix(2);
                mix(*x as u64 | (*y as u64) << 16);
                mix(*material as u64 | (*mass as u64) << 32);
            }
            ExternalEvent::Impact { cx, cy, radius } => {
                mix(3);
                mix(*cx as u64 | (*cy as u64) << 16 | (*radius as u64) << 32);
            }
            ExternalEvent::SeedOrganism { x, y } => {
                mix(4);
                mix(*x as u64 | (*y as u64) << 16);
            }
        }
    }
    mix(w.external_mass as u64);
    mix(w.next_id);
    mix(w.chem.table.len() as u64);
    h
}
