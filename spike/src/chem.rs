//! Seed-generated chemistry on an NK-style landscape.
//! Properties are unbounded reals ("raw"); physics and brains read the squashed
//! value in (0,1) ("props"). Each output property of a reaction is a random
//! Fourier function (a Gaussian-process sample) of K other properties drawn from
//! both reactants plus temperature. K is the ruggedness dial.
//! Material identity = quantized squashed property vector.

use crate::rng::Rng;
use std::collections::HashMap;

pub const NP: usize = 8;
pub const P_HARD: usize = 1;
pub const P_MELT: usize = 2;
pub const P_ENERGY: usize = 3;
pub const P_NUTRI: usize = 6;
pub const P_TOXIC: usize = 7;

pub type Props = [f32; NP];

pub const MAT_SOIL: u32 = 0;
pub const MAT_PLANT: u32 = 1;
pub const MAT_ROCK: u32 = 2;

pub const QUANT: f32 = 15.0;
const M: usize = 4; // Fourier terms per output property
const MAXV: usize = 2 * NP + 1; // max input dims: a-selection, b-selection, temperature
const TEMP_BUCKETS: f32 = 8.0;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Recipe {
    pub a: u32,
    pub b: u32,
    pub tq: u8,
}

pub struct Chemistry {
    pub seed: u64,
    pub n_base: usize,
    pub k: usize,
    pub raw: Vec<Props>,
    pub props: Vec<Props>,
    pub recipe: Vec<Option<Recipe>>,
    index: HashMap<Recipe, u32>,
    by_props: HashMap<[u8; NP], u32>,
    sel: Vec<Vec<u8>>,              // per output j: j itself plus K other property indices
    omega: Vec<[[f32; MAXV]; M]>,    // per output j, per term m: frequency vector
    phase: Vec<[f32; M]>,
    amp: Vec<[f32; M]>,
}

#[inline]
pub fn squash(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
#[inline]
pub fn logit(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}
#[inline]
pub fn quantize(p: &Props) -> [u8; NP] {
    let mut q = [0u8; NP];
    for i in 0..NP {
        q[i] = (p[i] * QUANT).round() as u8;
    }
    q
}
fn squash_all(r: &Props) -> Props {
    let mut p = [0f32; NP];
    for i in 0..NP {
        p[i] = squash(r[i]);
    }
    p
}

impl Chemistry {
    /// `k` in 0..=NP-1: how many other properties each output depends on.
    /// `amplitude`: size of the reaction term in log-units (1.0 moves 0.5 -> 0.73).
    /// `rho`: inverse lengthscale of the landscape (higher = more rugged per unit distance).
    pub fn generate(seed: u64, n_base: usize, k: usize, amplitude: f32, rho: f32) -> Self {
        assert!(n_base >= 4 && k < NP);
        let mut rng = Rng::new(seed ^ 0xC4E3_15A7_0000_0002);
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
                // Base materials in squashed range [0.05, 0.7]: headroom above them.
                p[j] = logit(0.05 + 0.65 * v);
            }
            raw.push(p);
        }
        raw[MAT_SOIL as usize][P_NUTRI] = -8.0;
        raw[MAT_SOIL as usize][P_TOXIC] = -8.0;
        raw[MAT_SOIL as usize][P_HARD] = logit(0.05);
        raw[MAT_PLANT as usize][P_NUTRI] = logit(0.25 + 0.2 * rng.f32());
        raw[MAT_PLANT as usize][P_HARD] = logit(0.05 + 0.15 * rng.f32());
        raw[MAT_PLANT as usize][P_TOXIC] = -8.0;
        raw[MAT_ROCK as usize][P_HARD] = logit(0.6 + 0.3 * rng.f32());
        raw[MAT_ROCK as usize][P_NUTRI] = -8.0;

        let mut sel = Vec::with_capacity(NP);
        let mut omega = Vec::with_capacity(NP);
        let mut phase = Vec::with_capacity(NP);
        let mut amp = Vec::with_capacity(NP);
        for j in 0..NP {
            let mut s = vec![j as u8];
            let mut pool: Vec<u8> = (0..NP as u8).filter(|&i| i as usize != j).collect();
            for _ in 0..k {
                let idx = rng.range(pool.len());
                s.push(pool.swap_remove(idx));
            }
            sel.push(s);
            let nv = 2 * (k + 1) + 1;
            let mut om = [[0f32; MAXV]; M];
            let mut ph = [0f32; M];
            let mut am = [0f32; M];
            for m in 0..M {
                for i in 0..nv {
                    om[m][i] = rng.normal() * rho;
                }
                ph[m] = rng.f32() * std::f32::consts::TAU;
                am[m] = rng.normal() * amplitude / (M as f32).sqrt();
            }
            omega.push(om);
            phase.push(ph);
            amp.push(am);
        }
        let props: Vec<Props> = raw.iter().map(squash_all).collect();
        let mut by_props = HashMap::new();
        for (i, p) in props.iter().enumerate() {
            by_props.entry(quantize(p)).or_insert(i as u32);
        }
        Chemistry { seed, n_base, k, raw, props, recipe: vec![None; n_base], index: HashMap::new(), by_props, sel, omega, phase, amp }
    }

    #[inline]
    pub fn is_artifact(&self, id: u32) -> bool {
        id as usize >= self.n_base
    }

    pub fn react(&self, a: &Props, b: &Props, temp: f32) -> Props {
        let mut out = [0f32; NP];
        let mut v = [0f32; MAXV];
        for j in 0..NP {
            let s = &self.sel[j];
            let n = s.len();
            for (i, &si) in s.iter().enumerate() {
                v[i] = a[si as usize];
                v[n + i] = b[si as usize];
            }
            v[2 * n] = (temp - 0.2) * 2.0;
            let nv = 2 * n + 1;
            let mut g = 0f32;
            for m in 0..M {
                let mut ph = self.phase[j][m];
                for i in 0..nv {
                    ph += self.omega[j][m][i] * v[i];
                }
                g += self.amp[j][m] * ph.cos();
            }
            out[j] = 0.5 * (a[j] + b[j]) + g;
        }
        out
    }

    /// Intern the combination of two materials at a temperature.
    pub fn combine(&mut self, a: u32, b: u32, temp: f32) -> u32 {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        let tq = (temp.clamp(0.0, 1.49) * TEMP_BUCKETS) as u8;
        let key = Recipe { a: lo, b: hi, tq };
        if let Some(&id) = self.index.get(&key) {
            return id;
        }
        let t = (tq as f32 + 0.5) / TEMP_BUCKETS;
        let r = self.react(&self.raw[lo as usize], &self.raw[hi as usize], t);
        let p = squash_all(&r);
        let q = quantize(&p);
        let id = match self.by_props.get(&q) {
            Some(&id) => id,
            None => {
                let id = self.props.len() as u32;
                self.raw.push(r);
                self.props.push(p);
                self.recipe.push(Some(key));
                self.by_props.insert(q, id);
                id
            }
        };
        self.index.insert(key, id);
        id
    }

    pub fn n_known(&self) -> usize {
        self.props.len()
    }
}
