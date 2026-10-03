//! Counter-based deterministic RNG. Every consumer derives its own generator
//! from (seed, tick, id), so thread count never changes history.

#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[inline]
pub fn hash3(a: u64, b: u64, c: u64) -> u64 {
    mix64(a ^ mix64(b.wrapping_add(0x9e37_79b9_7f4a_7c15) ^ mix64(c.wrapping_add(0x632b_e59b_d9b4_e019))))
}

#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: mix64(seed ^ 0x5851_f42d_4c95_7f2d) }
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.state)
    }
    /// Uniform in [0, 1).
    #[inline]
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    #[inline]
    pub fn range(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) as u128 * n as u128) >> 32) as usize
    }
    /// Approximately N(0,1): sum of 4 uniforms, rescaled. Bounded in [-3.46, 3.46].
    #[inline]
    pub fn normal(&mut self) -> f32 {
        let s = self.f32() + self.f32() + self.f32() + self.f32();
        (s - 2.0) * 1.732_050_8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn f32_in_unit_interval_and_range_in_bounds() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let f = r.f32();
            assert!((0.0..1.0).contains(&f));
            assert!(r.range(13) < 13);
        }
    }

    #[test]
    fn hash3_differs_on_each_argument() {
        let h = hash3(1, 2, 3);
        assert_ne!(h, hash3(2, 2, 3));
        assert_ne!(h, hash3(1, 3, 3));
        assert_ne!(h, hash3(1, 2, 4));
    }

    #[test]
    fn normal_has_zero_mean_unit_variance_roughly() {
        let mut r = Rng::new(3);
        let n = 100_000;
        let (mut s, mut s2) = (0f64, 0f64);
        for _ in 0..n {
            let x = r.normal() as f64;
            s += x;
            s2 += x * x;
        }
        let mean = s / n as f64;
        let var = s2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {}", mean);
        assert!((var - 1.0).abs() < 0.05, "var {}", var);
    }
}
