//! Counter-based deterministic RNG. No shared streams: every consumer derives
//! its own generator from (seed, tick, id), so thread count never changes history.

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

#[derive(Clone)]
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
        ((self.next_u64() >> 32) as usize * n) >> 32
    }
    /// Approximately N(0,1) via sum of 4 uniforms (cheap, bounded, deterministic).
    #[inline]
    pub fn normal(&mut self) -> f32 {
        let s = self.f32() + self.f32() + self.f32() + self.f32();
        (s - 2.0) * 1.732_050_8
    }
}
