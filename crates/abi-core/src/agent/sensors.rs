//! Observation vector. Everything is a property or a count; material ids never
//! reach the brain.

use super::brain::{NACT, NIN};
use super::{Agent, LIFESPAN, NO_ACTION, TAKE_MAX};
use crate::chem::generate::{MAT_CRUST, MAT_SOIL};
use crate::chem::props::*;
use crate::world::World;

pub fn observe(w: &World, a: &Agent, input: &mut [f32; NIN]) -> u8 {
    let mut k = 0;
    input[k] = a.energy / 300.0;
    k += 1;
    input[k] = a.age as f32 / LIFESPAN as f32;
    k += 1;
    for s in 0..2 {
        let (id, mass) = a.held[s];
        if mass > 0 {
            input[k..k + NP].copy_from_slice(w.chem.props(id));
        } else {
            input[k..k + NP].fill(0.0);
        }
        input[k + NP] = mass as f32 / TAKE_MAX as f32;
        k += NP + 1;
    }
    let here = w.grid.idx(a.x, a.y);
    let mut cells = [here; 5];
    for d in 0..4u8 {
        cells[d as usize + 1] = w.grid.neighbor(here, d);
    }
    for &c in &cells {
        let cell = &w.grid.cells[c];
        let (mut mn, mut mh, mut tot) = (0f32, 0f32, 0u32);
        for &(id, m) in &cell.inv {
            let p = w.chem.props(id);
            mn = mn.max(p[P_NUTRI]);
            mh = mh.max(p[P_HARD]);
            tot += m;
        }
        input[k] = mn;
        input[k + 1] = mh;
        input[k + 2] = (tot as f32 / 5000.0).min(2.0);
        input[k + 3] = cell.temp.min(2.0);
        k += 4;
    }
    let top = w.grid.cells[here].top2_non_soil();
    for s in 0..2 {
        if top[s].1 > 0 && top[s].0 != MAT_SOIL {
            input[k..k + NP].copy_from_slice(w.chem.props(top[s].0));
        } else {
            input[k..k + NP].fill(0.0);
        }
        k += NP;
    }
    input[k] = if w.grid.cells[here].bedrock > 0 { w.chem.props(MAT_CRUST)[P_HARD] } else { 0.0 };
    k += 1;
    let others = w.agents_in(here);
    input[k] = ((others.len() as f32 - 1.0).max(0.0) / 5.0).min(2.0);
    k += 1;
    let mut sig = [0f32; 2];
    let mut other_act = NO_ACTION;
    let mut mem = [0f32; 4];
    for o in others {
        if o.id != a.id {
            sig = o.signal;
            other_act = o.last_action;
            mem = a.memory.get(o.id);
            break;
        }
    }
    input[k] = sig[0];
    input[k + 1] = sig[1];
    k += 2;
    for i in 0..NACT {
        input[k + i] = if other_act as usize == i { 1.0 } else { 0.0 };
    }
    k += NACT;
    for i in 0..4 {
        input[k + i] = (mem[i] / 100.0).clamp(-1.0, 1.0);
    }
    k += 4;
    debug_assert_eq!(k, NIN);
    other_act
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::brain::NIN;
    use crate::agent::NO_ACTION;
    use crate::world::config::WorldConfig;
    use crate::world::World;

    fn small_world() -> World {
        World::new(&WorldConfig { seed: 1, width: 32, height: 32, pop0: 50, chem: crate::chem::generate::ChemParams { n_base: 12, ..Default::default() }, ..Default::default() })
    }

    #[test]
    fn observation_fills_every_input_with_finite_bounded_values() {
        let w = small_world();
        let a = &w.agents[0];
        let mut input = [f32::NAN; NIN];
        let other = observe(&w, a, &mut input);
        assert!(input.iter().all(|v| v.is_finite()));
        assert!(input.iter().all(|v| *v >= -1.0 && *v <= 2.0));
        assert!(other == NO_ACTION || other < 8);
        assert!((input[1] - 0.0).abs() < 1e-6, "age is zero at birth");
    }

    #[test]
    fn observation_sees_properties_not_ids() {
        // Two agents with identical surroundings but different held ids of identical props see the same input.
        let mut w = small_world();
        let (ida, idb) = (3u32, 3u32); // same material in both: inputs must match exactly
        w.agents[0].held[0] = (ida, 100);
        w.agents[1].held[0] = (idb, 100);
        w.agents[1].x = w.agents[0].x;
        w.agents[1].y = w.agents[0].y;
        w.agents[1].energy = w.agents[0].energy;
        w.sort_agents();
        let (a, b) = (w.agents[0].clone(), w.agents[1].clone());
        let (mut ia, mut ib) = ([0f32; NIN], [0f32; NIN]);
        observe(&w, &a, &mut ia);
        observe(&w, &b, &mut ib);
        assert_eq!(&ia[2..11], &ib[2..11]);
    }
}
