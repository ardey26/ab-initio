//! Seeded reaction function on an NK-style landscape. Each output property is
//! 0.5(a_j + b_j) plus a random Fourier function (a Gaussian-process sample) of
//! K other properties from both reactants and the temperature.

use super::props::*;
use crate::rng::Rng;
use serde::{Deserialize, Serialize};

const M: usize = 4; // Fourier terms per output
const MAXV: usize = 2 * NP + 1;

#[derive(Clone, Serialize, Deserialize)]
pub struct ReactionKernel {
    k: usize,
    sel: Vec<Vec<u8>>,           // per output j: j then K other indices
    omega: Vec<[[f32; MAXV]; M]>, // frequencies
    phase: Vec<[f32; M]>,
    amp: Vec<[f32; M]>,
}

impl ReactionKernel {
    pub fn generate(rng: &mut Rng, k: usize, amplitude: f32, rho: f32) -> Self {
        assert!(k < NP);
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
            sel.push(s);
            omega.push(om);
            phase.push(ph);
            amp.push(am);
        }
        ReactionKernel { k, sel, omega, phase, amp }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Raw props in, raw props out (clamped). Not symmetric in (a, b); callers
    /// canonicalize the order.
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
        clamp_raw(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn k_zero_has_no_cross_property_epistasis() {
        let kern = ReactionKernel::generate(&mut Rng::new(5), 0, 1.0, 1.0);
        let a = [0.1f32, -0.2, 0.3, 0.0, 0.5, -0.5, 0.2, -1.0];
        let mut b = [0.4f32, 0.1, -0.3, 0.2, 0.0, 0.3, -0.2, 0.5];
        let r1 = kern.react(&a, &b, 0.2);
        b[P_HARD] += 1.0;
        let r2 = kern.react(&a, &b, 0.2);
        for j in 0..NP {
            if j != P_HARD {
                assert!((r1[j] - r2[j]).abs() < 1e-6, "property {} changed under K=0", j);
            }
        }
        assert!((r1[P_HARD] - r2[P_HARD]).abs() > 1e-6);
    }

    #[test]
    fn k_seven_every_output_depends_on_every_input() {
        let kern = ReactionKernel::generate(&mut Rng::new(9), 7, 1.0, 1.0);
        let a = [0.0f32; NP];
        let b = [0.0f32; NP];
        let base = kern.react(&a, &b, 0.2);
        for i in 0..NP {
            let mut b2 = b;
            b2[i] += 1.0;
            let r = kern.react(&a, &b2, 0.2);
            let changed = (0..NP).filter(|&j| (r[j] - base[j]).abs() > 1e-6).count();
            assert!(changed >= NP - 1, "input {} changed only {} outputs", i, changed);
        }
    }

    #[test]
    fn output_is_clamped_and_temperature_matters() {
        let kern = ReactionKernel::generate(&mut Rng::new(1), 2, 3.0, 1.0);
        let a = [5.9f32; NP];
        let r = kern.react(&a, &a, 0.2);
        for v in r {
            assert!(v <= RAW_MAX && v >= RAW_MIN);
        }
        let cold = kern.react(&a, &[0.0; NP], 0.2);
        let hot = kern.react(&a, &[0.0; NP], 0.9);
        assert!(cold != hot);
    }

    #[test]
    fn same_seed_same_kernel() {
        let k1 = ReactionKernel::generate(&mut Rng::new(11), 3, 1.0, 1.0);
        let k2 = ReactionKernel::generate(&mut Rng::new(11), 3, 1.0, 1.0);
        let a = [0.3f32; NP];
        let b = [-0.3f32; NP];
        assert_eq!(k1.react(&a, &b, 0.5), k2.react(&a, &b, 0.5));
    }
}
