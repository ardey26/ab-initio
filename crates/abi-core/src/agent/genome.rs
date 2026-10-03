//! Genome: body, brain capacity, i8 weights, scalar genes. Mutation operates in
//! f32 space and requantizes with saturation.

use super::brain::*;
use crate::rng::{mix64, Rng};
use serde::{Deserialize, Serialize};

pub const SCALE0: f32 = 0.02;
const P_WEIGHT_MUT: f32 = 0.08;
const WEIGHT_SIGMA: f32 = 0.15;
const P_HIDDEN_MUT: f32 = 0.02;
const P_SCALAR_MUT: f32 = 0.2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    pub hidden: u8,
    pub w1: Vec<i8>,
    pub b1: Vec<i8>,
    pub w2: Vec<i8>,
    pub b2: Vec<i8>,
    pub scale: f32,
    pub body_size: f32,
    pub digest_thr: f32,
    pub eta: f32,
    pub imit: f32,
    pub explore: f32,
}

#[inline]
fn q(v: f32, scale: f32) -> i8 {
    (v / scale).round().clamp(-127.0, 127.0) as i8
}

impl Genome {
    pub fn random(rng: &mut Rng) -> Self {
        let hidden = NH_MIN + rng.range((NH_MAX - NH_MIN) as usize / 2) as u8; // start small: 4..18
        let h = hidden as usize;
        let mut gen = |n: usize, sigma: f32| -> Vec<i8> { (0..n).map(|_| q(rng.normal() * sigma, SCALE0)).collect() };
        let w1 = gen(h * NIN, 0.3);
        let b1 = gen(h, 0.1);
        let w2 = gen(NOUT * h, 0.3);
        let b2 = gen(NOUT, 0.1);
        Genome { hidden, w1, b1, w2, b2, scale: SCALE0, body_size: 0.8 + 0.4 * rng.f32(), digest_thr: 0.1 + 0.3 * rng.f32(), eta: 0.0, imit: 0.0, explore: 0.0 }
    }

    pub fn n_weights(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    pub fn bytes(&self) -> usize {
        self.n_weights() + 6 * 4 + 1
    }

    pub fn hash(&self) -> u64 {
        let mut h = mix64(self.hidden as u64 ^ 0xA5A5);
        for v in self.w1.iter().chain(&self.b1).chain(&self.w2).chain(&self.b2) {
            h = mix64(h ^ (*v as i64 as u64).wrapping_add(0x9e37_79b9_7f4a_7c15));
        }
        for f in [self.scale, self.body_size, self.digest_thr, self.eta, self.imit, self.explore] {
            h = mix64(h ^ f.to_bits() as u64);
        }
        h
    }

    pub fn mutate(&self, rng: &mut Rng) -> Self {
        let mut g = self.clone();
        let mutate_vec = |v: &mut Vec<i8>, rng: &mut Rng| {
            for x in v.iter_mut() {
                if rng.f32() < P_WEIGHT_MUT {
                    *x = q(*x as f32 * g.scale + rng.normal() * WEIGHT_SIGMA, g.scale);
                }
            }
        };
        mutate_vec(&mut g.w1, rng);
        mutate_vec(&mut g.b1, rng);
        mutate_vec(&mut g.w2, rng);
        mutate_vec(&mut g.b2, rng);
        if rng.f32() < P_HIDDEN_MUT {
            let grow = rng.f32() < 0.5;
            let h = g.hidden as usize;
            if grow && g.hidden < NH_MAX {
                g.hidden += 1;
                g.w1.extend(std::iter::repeat(0i8).take(NIN));
                g.b1.push(0);
                // w2 is [NOUT][h]: insert a zero column at the end of each row.
                let mut w2 = Vec::with_capacity(NOUT * (h + 1));
                for k in 0..NOUT {
                    w2.extend_from_slice(&g.w2[k * h..(k + 1) * h]);
                    w2.push(0);
                }
                g.w2 = w2;
            } else if !grow && g.hidden > NH_MIN {
                g.hidden -= 1;
                g.w1.truncate((h - 1) * NIN);
                g.b1.truncate(h - 1);
                let mut w2 = Vec::with_capacity(NOUT * (h - 1));
                for k in 0..NOUT {
                    w2.extend_from_slice(&g.w2[k * h..k * h + h - 1]);
                }
                g.w2 = w2;
            }
        }
        if rng.f32() < P_SCALAR_MUT {
            g.digest_thr = (g.digest_thr + rng.normal() * 0.05).clamp(0.0, 1.0);
        }
        if rng.f32() < P_SCALAR_MUT {
            g.body_size = (g.body_size + rng.normal() * 0.05).clamp(0.5, 2.0);
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn random_genome_has_consistent_shapes_and_budget() {
        let g = Genome::random(&mut Rng::new(1));
        assert!(g.hidden >= NH_MIN && g.hidden <= NH_MAX);
        assert_eq!(g.w1.len(), g.hidden as usize * NIN);
        assert_eq!(g.b1.len(), g.hidden as usize);
        assert_eq!(g.w2.len(), NOUT * g.hidden as usize);
        assert_eq!(g.b2.len(), NOUT);
        assert!(g.bytes() <= 2048 - 256, "genome {} bytes leaves no room for agent state", g.bytes());
        assert!(g.digest_thr >= 0.0 && g.digest_thr <= 1.0);
        assert!(g.body_size >= 0.5 && g.body_size <= 2.0);
    }

    #[test]
    fn mutation_changes_some_weights_and_keeps_shapes_valid() {
        let g = Genome::random(&mut Rng::new(2));
        let mut rng = Rng::new(3);
        let mut any_hidden_change = false;
        for _ in 0..200 {
            let m = g.mutate(&mut rng);
            assert_eq!(m.w1.len(), m.hidden as usize * NIN);
            assert_eq!(m.w2.len(), NOUT * m.hidden as usize);
            if m.hidden != g.hidden {
                any_hidden_change = true;
            }
        }
        assert!(any_hidden_change, "hidden size never mutated in 200 tries");
        let m = g.mutate(&mut Rng::new(4));
        let diff = g.w1.iter().zip(&m.w1).filter(|(a, b)| a != b).count();
        assert!(diff > 0 && diff < g.w1.len() / 2);
    }

    #[test]
    fn hash_is_stable_and_sensitive() {
        let g = Genome::random(&mut Rng::new(5));
        assert_eq!(g.hash(), g.clone().hash());
        let mut h = g.clone();
        h.w2[0] = h.w2[0].wrapping_add(1);
        assert_ne!(g.hash(), h.hash());
    }
}
