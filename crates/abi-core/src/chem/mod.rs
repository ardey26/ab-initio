pub mod generate;
pub mod kernel;
pub mod props;
pub mod table;

use generate::{generate_base, ChemParams};
use kernel::ReactionKernel;
use props::*;
use serde::{Deserialize, Serialize};
use table::{MaterialTable, Recipe};

pub const TEMP_BUCKETS: u8 = 4;
pub const HOT_TEMP: f32 = 0.3;

#[derive(Clone, Serialize, Deserialize)]
pub struct Chemistry {
    pub table: MaterialTable,
    pub kernel: ReactionKernel,
    pub params: ChemParams,
}

impl Chemistry {
    pub fn new(seed: u64, params: &ChemParams) -> Self {
        let base = generate_base(seed, params.n_base);
        let mut rng = crate::rng::Rng::new(seed ^ 0x4B45_524E_0000_0001);
        let kernel = ReactionKernel::generate(&mut rng, params.k, params.amplitude, params.rho);
        Chemistry { table: MaterialTable::new(base), kernel, params: *params }
    }

    #[inline]
    pub fn temp_bucket(temp: f32) -> u8 {
        ((temp.clamp(0.0, 1.499) * (TEMP_BUCKETS as f32) / 1.5) as u8).min(TEMP_BUCKETS - 1)
    }

    #[inline]
    pub fn hot(temp: f32) -> bool {
        temp > HOT_TEMP
    }

    /// Combine two materials at a temperature. Commutative by canonical ordering.
    pub fn combine(&mut self, a: MatId, b: MatId, temp: f32) -> MatId {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let tq = Self::temp_bucket(temp);
        let recipe = Recipe { a: lo, b: hi, tq };
        if let Some(id) = self.table.cached(&recipe) {
            return id;
        }
        let t = (tq as f32 + 0.5) * 1.5 / TEMP_BUCKETS as f32;
        let raw = self.kernel.react(self.table.raw(lo), self.table.raw(hi), t);
        self.table.intern(raw, recipe)
    }

    #[inline]
    pub fn props(&self, id: MatId) -> &Props {
        self.table.props(id)
    }
}
