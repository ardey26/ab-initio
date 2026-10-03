//! Fixed-size memory of interactions with specific others. Social structure is
//! whatever the observer finds in these tables.

use serde::{Deserialize, Serialize};

pub const SLOTS: usize = 8;
pub const VEC: usize = 4;
pub const EMPTY: u64 = u64::MAX;
const DECAY: f32 = 0.9;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SocialMemory {
    pub slots: [(u64, [f32; VEC], u32); SLOTS],
}

impl SocialMemory {
    pub fn new() -> Self {
        SocialMemory { slots: [(EMPTY, [0.0; VEC], 0); SLOTS] }
    }

    pub fn record(&mut self, other: u64, delta: [f32; VEC], tick: u32) {
        let slot = match self.slots.iter().position(|s| s.0 == other) {
            Some(i) => i,
            None => {
                let mut i = 0;
                for j in 1..SLOTS {
                    let (a, b) = (&self.slots[j], &self.slots[i]);
                    if a.0 == EMPTY && b.0 != EMPTY {
                        i = j;
                    } else if (a.0 == EMPTY) == (b.0 == EMPTY) && a.2 < b.2 {
                        i = j;
                    }
                }
                self.slots[i] = (other, [0.0; VEC], tick);
                i
            }
        };
        let s = &mut self.slots[slot];
        for k in 0..VEC {
            s.1[k] = s.1[k] * DECAY + delta[k];
        }
        s.2 = tick;
    }

    pub fn get(&self, other: u64) -> [f32; VEC] {
        self.slots.iter().find(|s| s.0 == other).map(|s| s.1).unwrap_or([0.0; VEC])
    }

    pub fn knows(&self, other: u64) -> bool {
        self.slots.iter().any(|s| s.0 == other)
    }
}

impl Default for SocialMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_accumulates_and_decays() {
        let mut m = SocialMemory::new();
        m.record(7, [1.0, 0.0, 0.0, 0.0], 1);
        m.record(7, [1.0, 0.0, 0.0, 0.0], 2);
        let v = m.get(7);
        assert!((v[0] - 1.9).abs() < 1e-5);
        assert_eq!(m.get(8), [0.0; VEC]);
        assert!(m.knows(7) && !m.knows(8));
    }

    #[test]
    fn replaces_least_recent_when_full() {
        let mut m = SocialMemory::new();
        for other in 0..SLOTS as u64 {
            m.record(other, [0.0, 0.0, 0.0, 1.0], other as u32 + 10);
        }
        m.record(99, [0.0, 0.0, 0.0, 1.0], 100);
        assert!(m.knows(99));
        assert!(!m.knows(0), "oldest entry evicted");
        assert!(m.knows(1));
    }
}
