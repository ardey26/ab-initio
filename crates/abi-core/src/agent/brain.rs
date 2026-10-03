//! Tiny MLP policy over i8 weights. Fixed input/output layout, evolvable hidden size.

use super::genome::Genome;
use crate::rng::Rng;

pub const NIN: usize = 72;
pub const NOUT: usize = 24;
pub const NACT: usize = 8;
pub const NH_MIN: u8 = 4;
pub const NH_MAX: u8 = 24;

pub const O_ACT: usize = 0; // 8 action logits
pub const O_DIR: usize = 8; // 4 direction logits
pub const O_TAKE: usize = 12; // 8 target property logits (squashed by consumer)
pub const O_DROP: usize = 20; // 2 slot logits
pub const O_EMIT: usize = 22; // 2 signal values (tanh by consumer)

pub fn forward(g: &Genome, input: &[f32; NIN], out: &mut [f32; NOUT]) {
    let h = g.hidden as usize;
    let s = g.scale;
    let mut hid = [0f32; NH_MAX as usize];
    for j in 0..h {
        let row = &g.w1[j * NIN..(j + 1) * NIN];
        let mut acc = g.b1[j] as f32 * s;
        for i in 0..NIN {
            acc += row[i] as f32 * s * input[i];
        }
        hid[j] = acc.tanh();
    }
    for k in 0..NOUT {
        let row = &g.w2[k * h..(k + 1) * h];
        let mut acc = g.b2[k] as f32 * s;
        for j in 0..h {
            acc += row[j] as f32 * s * hid[j];
        }
        out[k] = acc;
    }
}

pub fn softmax(logits: &[f32], out: &mut [f32]) {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for (o, &l) in out.iter_mut().zip(logits) {
        *o = (l - mx).exp();
        sum += *o;
    }
    for o in out.iter_mut() {
        *o /= sum;
    }
}

/// Gumbel-max sample from logits.
pub fn sample(logits: &[f32], rng: &mut Rng) -> usize {
    let mut best = 0;
    let mut bestv = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        let u = rng.f32().max(1e-7);
        let v = l - (-u.ln()).ln();
        if v > bestv {
            bestv = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::genome::Genome;
    use crate::rng::Rng;

    #[test]
    fn forward_is_deterministic_and_finite() {
        let g = Genome::random(&mut Rng::new(1));
        let input = [0.5f32; NIN];
        let mut a = [0f32; NOUT];
        let mut b = [0f32; NOUT];
        forward(&g, &input, &mut a);
        forward(&g, &input, &mut b);
        assert_eq!(a, b);
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn softmax_sums_to_one_and_sample_follows_logits() {
        let logits = [0.0f32, 0.0, 10.0, 0.0];
        let mut p = [0f32; 4];
        softmax(&logits, &mut p);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let mut rng = Rng::new(9);
        let hits = (0..1000).filter(|_| sample(&logits, &mut rng) == 2).count();
        assert!(hits > 990, "{}", hits);
    }
}
