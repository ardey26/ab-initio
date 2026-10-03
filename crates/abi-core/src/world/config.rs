use crate::chem::generate::ChemParams;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldConfig {
    pub seed: u64,
    pub width: usize,
    pub height: usize,
    pub chem: ChemParams,
    pub pop0: usize,
    pub max_agents: usize,
    /// Agents that grazing alone should support, per 1000 cells (spec R1).
    pub graze_capacity_per_1000: f32,
    pub volcanic_frac: f32,
    pub crust_frac: f32,
    pub deposit_frac: f32,
    pub checkpoint_every: u64,
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig {
            seed: 0,
            width: 256,
            height: 256,
            chem: ChemParams::default(),
            pop0: 2000,
            max_agents: 200_000,
            graze_capacity_per_1000: 100.0,
            volcanic_frac: 0.02,
            crust_frac: 0.3,
            deposit_frac: 0.03,
            checkpoint_every: 10_000,
        }
    }
}
