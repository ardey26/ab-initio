//! Per-window counters. Merged in chunk order so float sums are deterministic.
//! Set-like and map-like fields are bounded (bitsets, ordered map) so memory
//! does not grow with population or window length.

use crate::chem::props::MatId;
use std::collections::BTreeMap;

/// Material ids are lattice bins, always below 5^8.
const MAT_ID_LIMIT: usize = 390_625;
const MAT_WORDS: usize = 6104;
const BEHAVIOUR_WORDS: usize = 1024;

#[derive(Clone, Debug)]
pub struct Stats {
    pub agent_steps: u64,
    pub actions: [u64; 8],
    pub combines: u64,
    pub combines_hot: u64,
    pub strikes_mine: u64,
    pub strikes_hit: u64,
    pub strikes_split: u64,
    pub gives: u64,
    pub births: u64,
    pub deaths: u64,
    pub env_reactions: u64,
    pub env_hot: u64,
    pub fires: u64,
    pub energy_total: f64,
    pub energy_from_artifacts: f64,
    /// Bitset over material ids.
    pub materials_in_use: Box<[u64; MAT_WORDS]>,
    pub eaten_by_material: BTreeMap<MatId, f64>,
    /// Bitset over the 16-bit behaviour key.
    pub behaviours: Box<[u64; BEHAVIOUR_WORDS]>,
}

impl Default for Stats {
    fn default() -> Self {
        Stats {
            agent_steps: 0,
            actions: [0; 8],
            combines: 0,
            combines_hot: 0,
            strikes_mine: 0,
            strikes_hit: 0,
            strikes_split: 0,
            gives: 0,
            births: 0,
            deaths: 0,
            env_reactions: 0,
            env_hot: 0,
            fires: 0,
            energy_total: 0.0,
            energy_from_artifacts: 0.0,
            materials_in_use: Box::new([0; MAT_WORDS]),
            eaten_by_material: BTreeMap::new(),
            behaviours: Box::new([0; BEHAVIOUR_WORDS]),
        }
    }
}

fn or_into(dst: &mut [u64], src: &[u64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d |= *s;
    }
}

fn popcount(words: &[u64]) -> usize {
    words.iter().map(|w| w.count_ones() as usize).sum()
}

impl Stats {
    pub fn merge(&mut self, o: &Stats) {
        self.agent_steps += o.agent_steps;
        for i in 0..8 {
            self.actions[i] += o.actions[i];
        }
        self.combines += o.combines;
        self.combines_hot += o.combines_hot;
        self.strikes_mine += o.strikes_mine;
        self.strikes_hit += o.strikes_hit;
        self.strikes_split += o.strikes_split;
        self.gives += o.gives;
        self.births += o.births;
        self.deaths += o.deaths;
        self.env_reactions += o.env_reactions;
        self.env_hot += o.env_hot;
        self.fires += o.fires;
        self.energy_total += o.energy_total;
        self.energy_from_artifacts += o.energy_from_artifacts;
        or_into(&mut self.materials_in_use[..], &o.materials_in_use[..]);
        for (id, e) in &o.eaten_by_material {
            *self.eaten_by_material.entry(*id).or_insert(0.0) += *e;
        }
        or_into(&mut self.behaviours[..], &o.behaviours[..]);
    }

    pub fn note_material(&mut self, id: MatId) {
        debug_assert!((id as usize) < MAT_ID_LIMIT);
        self.materials_in_use[(id >> 6) as usize] |= 1u64 << (id & 63);
    }

    pub fn note_behaviour(&mut self, g: u32) {
        let g = g & 0xFFFF;
        self.behaviours[(g >> 6) as usize] |= 1u64 << (g & 63);
    }

    pub fn note_eaten(&mut self, id: MatId, energy: f64) {
        *self.eaten_by_material.entry(id).or_insert(0.0) += energy;
    }

    /// Distinct materials in use this window.
    pub fn distinct_in_use(&self) -> usize {
        popcount(&self.materials_in_use[..])
    }

    /// Distinct behaviour keys this window.
    pub fn distinct_behaviours(&self) -> usize {
        popcount(&self.behaviours[..])
    }

    /// Energy per material, sorted descending (ties by id ascending).
    pub fn top_eaten(&self, n: usize) -> Vec<(MatId, f64)> {
        let mut out: Vec<(MatId, f64)> = self.eaten_by_material.iter().map(|(k, v)| (*k, *v)).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(n);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_eaten_sums_per_material_and_sorts_descending() {
        let mut s = Stats::default();
        for (id, e) in [(4, 1.0), (2, 5.0), (4, 2.5), (7, 3.5)] {
            s.note_eaten(id, e);
        }
        assert_eq!(s.top_eaten(2), vec![(2, 5.0), (4, 3.5)]);
    }

    #[test]
    fn behaviour_bitset_dedups_and_merges() {
        let mut a = Stats::default();
        a.note_behaviour(0x1234);
        a.note_behaviour(0x1234);
        a.note_behaviour(0x1234);
        a.note_behaviour(0xFFFF);
        let mut b = Stats::default();
        b.note_behaviour(0x1234);
        b.note_behaviour(0x0001);
        a.merge(&b);
        assert_eq!(a.distinct_behaviours(), 3);
    }

    #[test]
    fn material_bitset_dedups() {
        let mut s = Stats::default();
        s.note_material(3);
        s.note_material(3);
        s.note_material(390_624);
        assert_eq!(s.distinct_in_use(), 2);
    }

    #[test]
    fn merge_sums_eaten_energy() {
        let mut a = Stats::default();
        a.note_eaten(7, 1.5);
        let mut b = Stats::default();
        b.note_eaten(7, 1.5);
        a.merge(&b);
        assert_eq!(a.top_eaten(1), vec![(7, 3.0)]);
    }
}
