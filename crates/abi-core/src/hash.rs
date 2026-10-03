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
    }
    for c in &w.grid.cells {
        let mut inv = c.inv.clone();
        inv.sort_unstable();
        for (id, m) in inv {
            mix(id as u64 | (m as u64) << 32);
        }
        mix(c.water as u64 | (c.bedrock as u64) << 32);
        mix(c.temp.to_bits() as u64);
    }
    mix(w.chem.table.len() as u64);
    h
}
