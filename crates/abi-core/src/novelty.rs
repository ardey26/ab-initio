//! Behaviour key: last 4 action codes conditioned on coarse bins of what is held.
//! Distinct keys in use per window is half of the health metric; distinct
//! materials in use is the other half.

pub fn gram(recent: u32, held_nutri_bin: u8, held_hard_bin: u8) -> u32 {
    (recent & 0xFFF) | ((held_nutri_bin as u32 & 3) << 12) | ((held_hard_bin as u32 & 3) << 14)
}

pub fn bin2(v: f32) -> u8 {
    (v.clamp(0.0, 0.999) * 4.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gram_packs_actions_and_bins_distinctly() {
        let a = gram(0o1234, 1, 2);
        let b = gram(0o1234, 2, 2);
        let c = gram(0o1233, 1, 2);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(gram(0o1234, 1, 2), a);
    }
}
