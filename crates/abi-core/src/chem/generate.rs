//! Base material generation. Three roles are guaranteed per seed: an inert soil,
//! a star-fed autotroph that is net-positive food, and a hard crust. Everything
//! else is seed-random with a seeded low-rank correlation structure.

use super::props::*;
use crate::rng::Rng;
use serde::{Deserialize, Serialize};

pub const MAT_SOIL: MatId = 0;
pub const MAT_AUTOTROPH: MatId = 1;
pub const MAT_CRUST: MatId = 2;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ChemParams {
    pub n_base: usize,
    pub k: usize,
    pub amplitude: f32,
    pub rho: f32,
}

impl Default for ChemParams {
    fn default() -> Self {
        ChemParams { n_base: 24, k: 2, amplitude: 1.0, rho: 1.0 }
    }
}

const NEVER: f32 = -8.0; // squashes to 0.0003

pub fn generate_base(seed: u64, n_base: usize) -> Vec<Props> {
    assert!(n_base >= 4);
    let mut rng = Rng::new(seed ^ 0xC4E3_15A7_0000_0003);
    let mut a = [[0f32; 3]; NP];
    let mut b = [0f32; NP];
    for j in 0..NP {
        for kk in 0..3 {
            a[j][kk] = rng.normal() * 1.2;
        }
        b[j] = rng.normal() * 0.5;
    }
    let mut raw = Vec::with_capacity(n_base);
    for _ in 0..n_base {
        let z = [rng.normal(), rng.normal(), rng.normal()];
        let mut p = [0f32; NP];
        for j in 0..NP {
            let v = squash(a[j][0] * z[0] + a[j][1] * z[1] + a[j][2] * z[2] + b[j]);
            p[j] = logit(0.05 + 0.65 * v); // squashed range [0.05, 0.70]
        }
        raw.push(p);
    }
    raw[MAT_SOIL as usize][P_NUTRI] = NEVER;
    raw[MAT_SOIL as usize][P_TOXIC] = NEVER;
    raw[MAT_SOIL as usize][P_HARD] = logit(0.05);
    raw[MAT_AUTOTROPH as usize][P_NUTRI] = logit(0.35 + 0.15 * rng.f32());
    raw[MAT_AUTOTROPH as usize][P_HARD] = logit(0.05 + 0.15 * rng.f32());
    raw[MAT_AUTOTROPH as usize][P_TOXIC] = NEVER;
    raw[MAT_CRUST as usize][P_HARD] = logit(0.6 + 0.3 * rng.f32());
    raw[MAT_CRUST as usize][P_NUTRI] = NEVER;
    raw
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_are_guaranteed_for_every_seed() {
        for seed in 0..200u64 {
            let base = generate_base(seed, 24);
            assert_eq!(base.len(), 24);
            let soil = squash_all(&base[MAT_SOIL as usize]);
            let auto = squash_all(&base[MAT_AUTOTROPH as usize]);
            let crust = squash_all(&base[MAT_CRUST as usize]);
            assert!(soil[P_NUTRI] < 0.01 && soil[P_TOXIC] < 0.01);
            assert!(auto[P_NUTRI] >= 0.35 && auto[P_NUTRI] <= 0.50, "seed {} autotroph nutrition {}", seed, auto[P_NUTRI]);
            assert!(auto[P_TOXIC] < 0.01);
            assert!(crust[P_HARD] >= 0.6);
            assert!(crust[P_NUTRI] < 0.01);
        }
    }

    #[test]
    fn other_materials_have_headroom() {
        let base = generate_base(3, 32);
        for p in base.iter().skip(3) {
            let s = squash_all(p);
            for v in s {
                assert!(v >= 0.049 && v <= 0.701, "{}", v);
            }
        }
    }
}
