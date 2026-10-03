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
        let hidden = NH_MIN + rng.range((NH_MAX - NH_MIN) as usize / 2) as u8; // start small: 4..=13
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

    fn grow(&mut self) {
        let h = self.hidden as usize;
        if self.hidden < NH_MAX {
            self.hidden += 1;
            self.w1.extend(std::iter::repeat(0i8).take(NIN));
            self.b1.push(0);
            // w2 is [NOUT][h]: insert a zero column at the end of each row.
            let mut w2 = Vec::with_capacity(NOUT * (h + 1));
            for k in 0..NOUT {
                w2.extend_from_slice(&self.w2[k * h..(k + 1) * h]);
                w2.push(0);
            }
            self.w2 = w2;
        }
    }

    fn shrink(&mut self) {
        let h = self.hidden as usize;
        if self.hidden > NH_MIN {
            self.hidden -= 1;
            self.w1.truncate((h - 1) * NIN);
            self.b1.truncate(h - 1);
            let mut w2 = Vec::with_capacity(NOUT * (h - 1));
            for k in 0..NOUT {
                w2.extend_from_slice(&self.w2[k * h..k * h + h - 1]);
            }
            self.w2 = w2;
        }
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
            if grow {
                g.grow();
            } else {
                g.shrink();
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
    use crate::agent::brain::{forward, NIN, NOUT};
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

    #[test]
    fn grow_preserves_forward_output_and_shrink_drops_last_column() {
        let g = Genome::random(&mut Rng::new(11));
        // Skip this test if genome is at boundary (can't grow/shrink further)
        let g = if g.hidden == NH_MIN || g.hidden == NH_MAX {
            let g2 = Genome::random(&mut Rng::new(7));
            if g2.hidden == NH_MIN || g2.hidden == NH_MAX {
                // Skip if still at boundary
                return;
            }
            g2
        } else {
            g
        };

        let h = g.hidden as usize;
        let input = [0.37f32; NIN];
        let mut before = [0f32; NOUT];
        forward(&g, &input, &mut before);

        // Test grow: add zero weights and bias, output should be unchanged
        let mut grown = g.clone();
        grown.grow();
        assert_eq!(
            grown.hidden as usize,
            h + 1,
            "hidden should increase by 1"
        );
        // Check w2 layout: each row of NOUT should have the old h values, then a 0
        for k in 0..NOUT {
            let old_row = &g.w2[k * h..(k + 1) * h];
            let new_row = &grown.w2[k * (h + 1)..(k + 1) * (h + 1)];
            for j in 0..h {
                assert_eq!(
                    new_row[j], old_row[j],
                    "grow: w2[{}, {}] should be unchanged",
                    k, j
                );
            }
            assert_eq!(
                new_row[h], 0,
                "grow: new w2 column should be zero"
            );
        }
        // Check new w1 and b1 are zero
        for i in 0..NIN {
            assert_eq!(grown.w1[h * NIN + i], 0, "grow: new w1 should be zero");
        }
        assert_eq!(grown.b1[h], 0, "grow: new b1 should be zero");

        // Forward pass should be unchanged (new neuron has zero input weights and bias)
        let mut after = [0f32; NOUT];
        forward(&grown, &input, &mut after);
        for k in 0..NOUT {
            let diff = (after[k] - before[k]).abs();
            assert!(
                diff < 1e-6,
                "grow: forward output[{}] changed by {}, expected < 1e-6",
                k, diff
            );
        }

        // Test shrink: remove last hidden neuron column from w2
        let mut shrunk = g.clone();
        shrunk.shrink();
        assert_eq!(
            shrunk.hidden as usize,
            h - 1,
            "hidden should decrease by 1"
        );
        // Check w2 layout: each row should have the first h-1 values of the old row
        for k in 0..NOUT {
            let old_row = &g.w2[k * h..(k + 1) * h];
            let new_row = &shrunk.w2[k * (h - 1)..(k + 1) * (h - 1)];
            for j in 0..h - 1 {
                assert_eq!(
                    new_row[j], old_row[j],
                    "shrink: w2[{}, {}] should be old row's first {} values",
                    k, j, h - 1
                );
            }
        }
        // Check truncated w1 and b1
        assert_eq!(
            shrunk.w1.len(),
            (h - 1) * NIN,
            "shrink: w1 length should be (h-1)*NIN"
        );
        assert_eq!(shrunk.b1.len(), h - 1, "shrink: b1 length should be h-1");
    }
}
