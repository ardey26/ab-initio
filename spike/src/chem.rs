//! Seed-generated chemistry. Materials are property vectors. Combination is a
//! seed-generated function of two property vectors and temperature. Artifacts are
//! interned composition nodes, so nesting is unbounded but storage is bounded.

use crate::rng::Rng;
use std::collections::HashMap;

pub const NP: usize = 8;
pub const P_DENSITY: usize = 0;
pub const P_HARD: usize = 1;
pub const P_MELT: usize = 2;
pub const P_ENERGY: usize = 3;
pub const P_BRITTLE: usize = 4;
pub const P_SOLUBLE: usize = 5;
pub const P_NUTRI: usize = 6;
pub const P_TOXIC: usize = 7;
pub const PROP_NAMES: [&str; NP] = ["density", "hard", "melt", "energy", "brittle", "soluble", "nutri", "toxic"];

pub type Props = [f32; NP];

/// Roles the generator guarantees exist (a star-fed autotroph material and a hard
/// crust). Everything else about them is seed-random.
pub const MAT_SOIL: u32 = 0;
pub const MAT_PLANT: u32 = 1;
pub const MAT_ROCK: u32 = 2;

const NF: usize = 5 * NP; // symmetric pair features: prod, absdiff, max, min, sum
const TERMS: usize = 4; // sparse terms per output property
const COLD_SCALE: f32 = 0.3; // mechanical mixing reacts weakly
const HOT_SCALE: f32 = 1.0;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Recipe {
    pub a: u32,
    pub b: u32,
    pub hot: bool,
}

pub struct Chemistry {
    pub seed: u64,
    pub n_base: usize,
    pub props: Vec<Props>,
    pub recipe: Vec<Option<Recipe>>,
    index: HashMap<Recipe, u32>,
    kernel: [[(u8, f32); TERMS]; NP],
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Chemistry {
    pub fn generate(seed: u64, n_base: usize) -> Self {
        assert!(n_base >= 4);
        let mut rng = Rng::new(seed ^ 0xC4E3_15A7_0000_0001);
        // Seed-specific correlation structure: props = sigmoid(A z + b), z in R^3.
        let mut a = [[0f32; 3]; NP];
        let mut b = [0f32; NP];
        for j in 0..NP {
            for k in 0..3 {
                a[j][k] = rng.normal() * 1.2;
            }
            b[j] = rng.normal() * 0.5;
        }
        let mut props = Vec::with_capacity(n_base);
        for _ in 0..n_base {
            let z = [rng.normal(), rng.normal(), rng.normal()];
            let mut p = [0f32; NP];
            for j in 0..NP {
                let v = sigmoid(a[j][0] * z[0] + a[j][1] * z[1] + a[j][2] * z[2] + b[j]);
                // Base materials live in [0.05, 0.7]: headroom exists for combinations.
                p[j] = 0.05 + 0.65 * v;
            }
            props.push(p);
        }
        // Roles.
        props[MAT_SOIL as usize][P_NUTRI] = 0.0;
        props[MAT_SOIL as usize][P_TOXIC] = 0.0;
        props[MAT_SOIL as usize][P_HARD] = 0.05;
        props[MAT_PLANT as usize][P_NUTRI] = 0.25 + 0.2 * rng.f32();
        props[MAT_PLANT as usize][P_HARD] = 0.05 + 0.15 * rng.f32();
        props[MAT_PLANT as usize][P_TOXIC] = 0.0;
        props[MAT_ROCK as usize][P_HARD] = 0.6 + 0.3 * rng.f32();
        props[MAT_ROCK as usize][P_NUTRI] = 0.0;

        let mut kernel = [[(0u8, 0f32); TERMS]; NP];
        for j in 0..NP {
            for t in 0..TERMS {
                kernel[j][t] = (rng.range(NF) as u8, rng.normal() * 0.5);
            }
        }
        let recipe = vec![None; n_base];
        Chemistry { seed, n_base, props, recipe, index: HashMap::new(), kernel }
    }

    #[inline]
    pub fn is_artifact(&self, id: u32) -> bool {
        id as usize >= self.n_base
    }

    /// Temperature at which the pair reacts fully.
    #[inline]
    pub fn hot_threshold(a: &Props, b: &Props) -> f32 {
        a[P_MELT].min(b[P_MELT])
    }

    pub fn react(&self, a: &Props, b: &Props, hot: bool) -> Props {
        let mut f = [0f32; NF];
        for i in 0..NP {
            f[i] = a[i] * b[i];
            f[NP + i] = (a[i] - b[i]).abs();
            f[2 * NP + i] = a[i].max(b[i]);
            f[3 * NP + i] = a[i].min(b[i]);
            f[4 * NP + i] = a[i] + b[i];
        }
        let scale = if hot { HOT_SCALE } else { COLD_SCALE };
        let mut out = [0f32; NP];
        for j in 0..NP {
            let mut r = 0f32;
            for &(k, w) in &self.kernel[j] {
                r += w * f[k as usize];
            }
            out[j] = (0.5 * (a[j] + b[j]) + scale * r).clamp(0.0, 1.0);
        }
        out
    }

    /// Intern the combination of two material/artifact ids at a temperature.
    pub fn combine(&mut self, a: u32, b: u32, temp: f32) -> u32 {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let hot = temp >= Self::hot_threshold(&self.props[lo as usize], &self.props[hi as usize]);
        let key = Recipe { a: lo, b: hi, hot };
        if let Some(&id) = self.index.get(&key) {
            return id;
        }
        let p = self.react(&self.props[lo as usize], &self.props[hi as usize], hot);
        let id = self.props.len() as u32;
        self.props.push(p);
        self.recipe.push(Some(key));
        self.index.insert(key, id);
        id
    }

    pub fn n_known(&self) -> usize {
        self.props.len()
    }
}
