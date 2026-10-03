//! Tiny MLP policy. Fixed topology for the spike; the full system evolves dims.

use crate::rng::Rng;

pub const NIN: usize = 60;
pub const NH: usize = 16;
pub const NOUT: usize = 24;
pub const NW: usize = NIN * NH + NH + NH * NOUT + NOUT;

// Output layout.
pub const O_ACT: usize = 0; // 8 action logits
pub const O_DIR: usize = 8; // 4 move direction logits
pub const O_TAKE: usize = 12; // 8: target property vector (sigmoid) for take
pub const O_DROP: usize = 20; // 2: drop slot 0 vs 1
pub const O_EMIT: usize = 22; // 2 signal values

#[derive(Clone)]
pub struct Genome {
    pub w: Box<[f32; NW]>,
    pub digest_thr: f32,
}

impl Genome {
    pub fn random(rng: &mut Rng) -> Self {
        let mut w = Box::new([0f32; NW]);
        for x in w.iter_mut() {
            *x = rng.normal() * 0.3;
        }
        Genome { w, digest_thr: 0.1 + 0.3 * rng.f32() }
    }

    pub fn mutate(&self, rng: &mut Rng) -> Self {
        let mut g = self.clone();
        for x in g.w.iter_mut() {
            if rng.f32() < 0.08 {
                *x += rng.normal() * 0.15;
            }
        }
        if rng.f32() < 0.2 {
            g.digest_thr = (g.digest_thr + rng.normal() * 0.05).clamp(0.0, 1.0);
        }
        g
    }

    #[inline]
    pub fn forward(&self, input: &[f32; NIN], out: &mut [f32; NOUT]) {
        let w = &self.w;
        let mut h = [0f32; NH];
        let (w1, rest) = w.split_at(NIN * NH);
        let (b1, rest) = rest.split_at(NH);
        let (w2, b2) = rest.split_at(NH * NOUT);
        for j in 0..NH {
            let row = &w1[j * NIN..(j + 1) * NIN];
            let mut s = b1[j];
            for i in 0..NIN {
                s += row[i] * input[i];
            }
            h[j] = s.tanh();
        }
        for k in 0..NOUT {
            let row = &w2[k * NH..(k + 1) * NH];
            let mut s = b2[k];
            for j in 0..NH {
                s += row[j] * h[j];
            }
            out[k] = s;
        }
    }
}

/// Sample an index from logits with a Gumbel-max draw (stochastic policy).
#[inline]
pub fn sample(logits: &[f32], rng: &mut Rng) -> usize {
    let mut best = 0;
    let mut bestv = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        let u = rng.f32().max(1e-7);
        let g = -(-u.ln()).ln();
        let v = l + g;
        if v > bestv {
            bestv = v;
            best = i;
        }
    }
    best
}
