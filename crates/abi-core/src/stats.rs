//! Per-window counters. Merged in chunk order so float sums are deterministic.

use crate::chem::props::MatId;

#[derive(Clone, Default, Debug)]
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
    pub materials_in_use: Vec<MatId>,
    pub eaten_by_material: Vec<(MatId, f64)>,
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
        self.materials_in_use.extend_from_slice(&o.materials_in_use);
        self.eaten_by_material.extend_from_slice(&o.eaten_by_material);
    }

    /// Distinct materials in use this window.
    pub fn distinct_in_use(&self) -> usize {
        let mut v = self.materials_in_use.clone();
        v.sort_unstable();
        v.dedup();
        v.len()
    }

    /// Energy per material, summed and sorted descending.
    pub fn top_foods(&self, n: usize) -> Vec<(MatId, f64)> {
        let mut v = self.eaten_by_material.clone();
        v.sort_by_key(|e| e.0);
        let mut out: Vec<(MatId, f64)> = Vec::new();
        for (id, e) in v {
            match out.last_mut() {
                Some(l) if l.0 == id => l.1 += e,
                _ => out.push((id, e)),
            }
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        out.truncate(n);
        out
    }
}
