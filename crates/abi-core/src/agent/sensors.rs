//! Observation vector. Everything is a property or a count; material ids never
//! reach the brain.

use super::brain::{NACT, NIN};
use super::{Agent, LIFESPAN, NO_ACTION, TAKE_MAX};
use crate::chem::generate::{MAT_CRUST, MAT_SOIL};
use crate::chem::props::*;
use crate::world::World;

pub fn observe(w: &World, a: &Agent, input: &mut [f32; NIN]) -> u8 {
    let mut k = 0;
    input[k] = (a.energy / 300.0).clamp(0.0, 2.0);
    k += 1;
    input[k] = (a.age as f32 / LIFESPAN as f32).clamp(0.0, 2.0);
    k += 1;
    for s in 0..2 {
        let (id, mass) = a.held[s];
        if mass > 0 {
            input[k..k + NP].copy_from_slice(w.chem.props(id));
        } else {
            input[k..k + NP].fill(0.0);
        }
        input[k + NP] = (mass as f32 / TAKE_MAX as f32).clamp(0.0, 2.0);
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
            sig = [o.signal[0].clamp(-1.0, 1.0), o.signal[1].clamp(-1.0, 1.0)];
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
        // The held block carries the material's properties and mass, never its id.
        let mut w = small_world();
        let (ida, idb) = (3u32, 5u32);
        assert_ne!(w.chem.props(ida), w.chem.props(idb), "test needs materials with different props");
        let pa = w.agents.iter().position(|a| a.id == 0).unwrap();
        let pb = w.agents.iter().position(|a| a.id == 1).unwrap();
        w.agents[pa].held[0] = (ida, 100);
        w.agents[pb].held[0] = (idb, 100);
        w.agents[pb].x = w.agents[pa].x;
        w.agents[pb].y = w.agents[pa].y;
        w.sort_agents();
        let a = w.agents.iter().find(|a| a.id == 0).unwrap().clone();
        let b = w.agents.iter().find(|a| a.id == 1).unwrap().clone();
        assert_eq!((a.x, a.y), (b.x, b.y));
        let (mut ia, mut ib) = ([0f32; NIN], [0f32; NIN]);
        observe(&w, &a, &mut ia);
        observe(&w, &b, &mut ib);
        assert_eq!(&ia[2..10], w.chem.props(ida));
        assert_eq!(&ib[2..10], w.chem.props(idb));
        assert!((ia[10] - 0.1).abs() < 1e-6 && (ib[10] - 0.1).abs() < 1e-6);
        for v in ia.iter() {
            assert!((v - ida as f32).abs() > 1e-6, "id {ida} leaked into the observation");
        }
        for v in ib.iter() {
            assert!((v - idb as f32).abs() > 1e-6, "id {idb} leaked into the observation");
        }
    }

    #[test]
    fn observation_is_bounded_for_extreme_agent_state() {
        let mut w = small_world();
        let me = w.agents.iter().position(|a| a.id == 0).unwrap();
        w.agents[me].energy = 1.0e6;
        w.agents[me].age = 1_000_000;
        w.agents[me].held[0] = (3, 1_000_000);
        let a = w.agents[me].clone();
        let mut input = [0f32; NIN];
        observe(&w, &a, &mut input);
        assert!(input.iter().all(|v| v.is_finite() && *v >= -1.0 && *v <= 2.0));
        w.agents[me].energy = -50.0;
        let a = w.agents[me].clone();
        observe(&w, &a, &mut input);
        assert!(input.iter().all(|v| v.is_finite() && *v >= -1.0 && *v <= 2.0));
    }

    #[test]
    fn observation_clamps_neighbour_signal() {
        let mut w = small_world();
        let other = w.agents.iter().position(|a| a.id == 1).unwrap();
        let me = w.agents.iter().position(|a| a.id == 0).unwrap();
        w.agents[other].x = w.agents[me].x;
        w.agents[other].y = w.agents[me].y;
        w.agents[other].signal = [50.0, -50.0];
        w.sort_agents();
        let a = w.agents.iter().find(|a| a.id == 0).unwrap().clone();
        let mut input = [0f32; NIN];
        observe(&w, &a, &mut input);
        assert_eq!(&input[58..60], &[1.0, -1.0]);
    }

    #[test]
    fn observe_sees_lowest_id_neighbour() {
        let mut w = World::new(&WorldConfig { seed: 1, width: 32, height: 32, pop0: 3, chem: crate::chem::generate::ChemParams { n_base: 12, ..Default::default() }, ..Default::default() });
        let at = |w: &World, id: u64| w.agents.iter().position(|a| a.id == id).unwrap();
        let (x, y) = (w.agents[at(&w, 0)].x, w.agents[at(&w, 0)].y);
        for id in [1u64, 2] {
            let i = at(&w, id);
            w.agents[i].x = x;
            w.agents[i].y = y;
        }
        let i1 = at(&w, 1);
        let i2 = at(&w, 2);
        w.agents[i1].last_action = 5;
        w.agents[i2].last_action = 6;
        w.sort_agents();
        let a = w.agents.iter().find(|a| a.id == 0).unwrap().clone();
        let mut input = [0f32; NIN];
        let other = observe(&w, &a, &mut input);
        assert_eq!(other, 5);
        for i in 0..8 {
            assert_eq!(input[60 + i], if i == 5 { 1.0 } else { 0.0 }, "one-hot index {i}");
        }
        assert_eq!(input[57], 2.0 / 5.0);
    }
}
