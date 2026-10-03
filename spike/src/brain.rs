//! Tiny MLP policy with a genetic initial network and a lifetime copy whose
//! action head learns by reward-modulated eligibility traces.

use crate::rng::Rng;

pub const NIN: usize = 68;
pub const NH: usize = 16;
pub const NOUT: usize = 24;
pub const NW: usize = NIN * NH + NH + NH * NOUT + NOUT;
pub const NACT: usize = 8;

pub const O_ACT: usize = 0; // 8 action logits
pub const O_DIR: usize = 8; // 4 move direction logits
pub const O_TAKE: usize = 12; // 8: target property vector (sigmoid) for take
pub const O_DROP: usize = 20; // 2: drop slot 0 vs 1
pub const O_EMIT: usize = 22; // 2 signal values

const W2_OFF: usize = NIN * NH + NH; // start of output weights
const B2_OFF: usize = W2_OFF + NH * NOUT;
pub const WCLAMP: f32 = 4.0;
pub const TRACE_DECAY: f32 = 0.9;

#[derive(Clone)]
pub struct Genome {
    pub w: Box<[f32; NW]>,
    pub digest_thr: f32,
    pub eta: f32,  // lifetime learning rate (0 = no learning)
    pub imit: f32, // imitation gain
}

impl Genome {
    pub fn random(rng: &mut Rng, learn: bool) -> Self {
        let mut w = Box::new([0f32; NW]);
        for x in w.iter_mut() {
            *x = rng.normal() * 0.3;
        }
        let (eta, imit) = if learn { (0.005 * rng.f32(), 0.002 * rng.f32()) } else { (0.0, 0.0) };
        Genome { w, digest_thr: 0.1 + 0.3 * rng.f32(), eta, imit }
    }

    pub fn mutate(&self, rng: &mut Rng, learn: bool) -> Self {
        let mut g = self.clone();
        for x in g.w.iter_mut() {
            if rng.f32() < 0.08 {
                *x += rng.normal() * 0.15;
            }
        }
        if rng.f32() < 0.2 {
            g.digest_thr = (g.digest_thr + rng.normal() * 0.05).clamp(0.0, 1.0);
        }
        if learn {
            if rng.f32() < 0.2 {
                g.eta = (g.eta + rng.normal() * 0.002).clamp(0.0, 0.05);
            }
            if rng.f32() < 0.2 {
                g.imit = (g.imit + rng.normal() * 0.001).clamp(0.0, 0.02);
            }
        }
        g
    }
}

#[inline]
pub fn forward(w: &[f32; NW], input: &[f32; NIN], h: &mut [f32; NH], out: &mut [f32; NOUT]) {
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

#[inline]
pub fn softmax(logits: &[f32], out: &mut [f32]) {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0f32;
    for (o, &l) in out.iter_mut().zip(logits) {
        *o = (l - mx).exp();
        s += *o;
    }
    for o in out.iter_mut() {
        *o /= s;
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

/// Lifetime learning state: eligibility trace over the action head, reward baseline.
#[derive(Clone)]
pub struct Learner {
    pub trace: Box<[f32; NH * NACT]>,
    pub trace_b: [f32; NACT],
    pub r_mean: f32,
}

impl Learner {
    pub fn new() -> Self {
        Learner { trace: Box::new([0f32; NH * NACT]), trace_b: [0f32; NACT], r_mean: 0.0 }
    }

    /// Accumulate the policy-gradient direction for the action taken in state h.
    pub fn accumulate(&mut self, h: &[f32; NH], probs: &[f32; NACT], act: usize) {
        for k in 0..NACT {
            let d = (if k == act { 1.0 } else { 0.0 }) - probs[k];
            self.trace_b[k] = self.trace_b[k] * TRACE_DECAY + d;
            for j in 0..NH {
                let t = &mut self.trace[k * NH + j];
                *t = *t * TRACE_DECAY + d * h[j];
            }
        }
    }

    /// Apply reward to the action head of the lifetime weights.
    pub fn reward(&mut self, w: &mut [f32; NW], delta_e: f32, eta: f32) {
        let r = delta_e - self.r_mean;
        self.r_mean += 0.02 * (delta_e - self.r_mean);
        if eta == 0.0 {
            return;
        }
        let s = eta * r;
        for k in 0..NACT {
            w[B2_OFF + k] = (w[B2_OFF + k] + s * self.trace_b[k]).clamp(-WCLAMP, WCLAMP);
            for j in 0..NH {
                let i = W2_OFF + k * NH + j;
                w[i] = (w[i] + s * self.trace[k * NH + j]).clamp(-WCLAMP, WCLAMP);
            }
        }
    }
}

/// Nudge the action head toward an observed action in the current state.
pub fn imitate(w: &mut [f32; NW], h: &[f32; NH], probs: &[f32; NACT], other_act: usize, gain: f32) {
    if gain == 0.0 {
        return;
    }
    for k in 0..NACT {
        let d = gain * ((if k == other_act { 1.0 } else { 0.0 }) - probs[k]);
        w[B2_OFF + k] = (w[B2_OFF + k] + d).clamp(-WCLAMP, WCLAMP);
        for j in 0..NH {
            let i = W2_OFF + k * NH + j;
            w[i] = (w[i] + d * h[j]).clamp(-WCLAMP, WCLAMP);
        }
    }
}
