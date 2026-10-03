//! Material properties. Raw values are unbounded reals clamped to [RAW_MIN, RAW_MAX];
//! physics and brains read the squashed value in (0, 1). Identity is the quantized
//! squashed vector.

pub const NP: usize = 8;
pub type Props = [f32; NP];
pub type MatId = u32;
pub type BinKey = [u8; NP];

pub const P_DENSITY: usize = 0;
pub const P_HARD: usize = 1;
pub const P_MELT: usize = 2;
pub const P_ENERGY: usize = 3;
pub const P_BRITTLE: usize = 4;
pub const P_SOLUBLE: usize = 5;
pub const P_NUTRI: usize = 6;
pub const P_TOXIC: usize = 7;
pub const PROP_NAMES: [&str; NP] = ["density", "hardness", "melting", "energy", "brittleness", "solubility", "nutrition", "toxicity"];

pub const RAW_MIN: f32 = -6.0;
pub const RAW_MAX: f32 = 6.0;
pub const Q_LEVELS: u8 = 5;

#[inline]
pub fn squash(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn logit(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

pub fn clamp_raw(r: &Props) -> Props {
    let mut o = *r;
    for v in o.iter_mut() {
        *v = v.clamp(RAW_MIN, RAW_MAX);
    }
    o
}

pub fn squash_all(r: &Props) -> Props {
    let mut o = [0f32; NP];
    for i in 0..NP {
        o[i] = squash(r[i]);
    }
    o
}

/// Quantize squashed props to Q_LEVELS levels per property (0 ..= Q_LEVELS-1).
pub fn quantize(p: &Props) -> BinKey {
    let mut q = [0u8; NP];
    let scale = (Q_LEVELS - 1) as f32;
    for i in 0..NP {
        q[i] = (p[i].clamp(0.0, 1.0) * scale).round() as u8;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squash_and_logit_invert() {
        for p in [0.05f32, 0.3, 0.5, 0.9] {
            assert!((squash(logit(p)) - p).abs() < 1e-5);
        }
    }

    #[test]
    fn clamp_raw_bounds_every_property() {
        let r = clamp_raw(&[-100.0, 100.0, 0.0, 7.0, -7.0, 6.0, -6.0, 1.0]);
        assert_eq!(r, [RAW_MIN, RAW_MAX, 0.0, RAW_MAX, RAW_MIN, RAW_MAX, RAW_MIN, 1.0]);
    }

    #[test]
    fn quantize_uses_q_levels_inclusive() {
        let lo = quantize(&[0.0; NP]);
        let hi = quantize(&[1.0; NP]);
        assert_eq!(lo, [0; NP]);
        assert_eq!(hi, [Q_LEVELS - 1; NP]);
        let mid = quantize(&[0.5; NP]);
        assert_eq!(mid, [2; NP]);
    }

    #[test]
    fn lattice_size_is_bounded() {
        assert_eq!((Q_LEVELS as u64).pow(NP as u32), 390_625);
    }
}
