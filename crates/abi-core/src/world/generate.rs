//! Seeded terrain: periodic heightfield and wetness from harmonic sums, volcanic
//! cells, crust with ore, surface deposits, and the autotroph growth rate
//! calibrated so grazing alone supports a bounded population (spec R1).

use super::cell::Cell;
use super::config::WorldConfig;
use super::grid::Grid;
use crate::chem::generate::{MAT_AUTOTROPH, MAT_SOIL};
use crate::chem::props::MatId;
use crate::rng::{hash3, Rng};

pub const SOIL0: u32 = 5000;
pub const AUTOTROPH0: u32 = 1000;
pub const BEDROCK0: u32 = 1_000_000;
pub const DEPOSIT0: u32 = 2000;
pub const AMBIENT: f32 = 0.2;
pub const VOLCANIC_TEMP: f32 = 0.8;
pub const WATER_MAX: u32 = 2000;
/// Autotroph mass one grazing agent consumes per tick at equilibrium (spike 4 measurement).
pub const GRAZE_MASS_PER_AGENT_TICK: f32 = 19.0;

pub struct Terrain {
    pub grid: Grid,
    pub growth_per_tick: u32,
    pub fertile_cells: usize,
}

struct Harmonics {
    terms: Vec<(f32, f32, f32, f32)>, // (kx, ky, phase, amp)
}

impl Harmonics {
    fn new(rng: &mut Rng, width: usize, height: usize, n: usize) -> Self {
        let mut terms = Vec::with_capacity(n);
        for i in 0..n {
            let octave = 1 + (i / 2) as i32;
            let kx = (rng.range(2 * octave as usize) as i32 + 1) as f32 * std::f32::consts::TAU / width as f32;
            let ky = (rng.range(2 * octave as usize) as i32 + 1) as f32 * std::f32::consts::TAU / height as f32;
            let sx = if rng.f32() < 0.5 { 1.0 } else { -1.0 };
            terms.push((kx * sx, ky, rng.f32() * std::f32::consts::TAU, 1.0 / octave as f32));
        }
        Harmonics { terms }
    }
    /// Value in [0, 1].
    fn at(&self, x: f32, y: f32) -> f32 {
        let (mut s, mut norm) = (0f32, 0f32);
        for &(kx, ky, ph, amp) in &self.terms {
            s += amp * (kx * x + ky * y + ph).sin();
            norm += amp;
        }
        (0.5 + 0.5 * s / norm).clamp(0.0, 1.0)
    }
}

pub fn calibrate_growth(cells: usize, fertile_cells: usize, capacity_per_1000: f32) -> u32 {
    if fertile_cells == 0 {
        return 0;
    }
    let target_agents = capacity_per_1000 * cells as f32 / 1000.0;
    let total_growth = target_agents * GRAZE_MASS_PER_AGENT_TICK;
    (total_growth / fertile_cells as f32).round().max(1.0) as u32
}

pub fn generate_terrain(cfg: &WorldConfig) -> Terrain {
    let mut grid = Grid::new(cfg.width, cfg.height);
    let mut rng = Rng::new(cfg.seed ^ 0x5445_5252_0000_0001);
    let height_h = Harmonics::new(&mut rng, cfg.width, cfg.height, 6);
    let wet_h = Harmonics::new(&mut rng, cfg.width, cfg.height, 6);
    let n_base = cfg.chem.n_base;
    for idx in 0..grid.len() {
        let (x, y) = grid.xy(idx);
        let mut r = Rng::new(hash3(cfg.seed ^ 0x4345_4C4C, idx as u64, 0));
        let c: &mut Cell = &mut grid.cells[idx];
        c.elevation = height_h.at(x as f32, y as f32);
        let wet = wet_h.at(x as f32, y as f32);
        c.water = (wet * WATER_MAX as f32) as u32;
        c.ambient = if r.f32() < cfg.volcanic_frac { VOLCANIC_TEMP } else { AMBIENT };
        c.temp = c.ambient;
        c.add(MAT_SOIL, SOIL0);
        if c.fertile() {
            c.add(MAT_AUTOTROPH, AUTOTROPH0);
        }
        if r.f32() < cfg.crust_frac {
            c.bedrock = BEDROCK0;
            c.ore = 3 + r.range(n_base - 3) as MatId;
        }
        for m in 3..n_base as MatId {
            if r.f32() < cfg.deposit_frac {
                c.add(m, DEPOSIT0);
            }
        }
    }
    let fertile_cells = grid.cells.iter().filter(|c| c.fertile()).count();
    let growth_per_tick = calibrate_growth(grid.len(), fertile_cells, cfg.graze_capacity_per_1000);
    Terrain { grid, growth_per_tick, fertile_cells }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chem::generate::{MAT_AUTOTROPH, MAT_SOIL};
    use crate::world::config::WorldConfig;

    #[test]
    fn calibration_scales_with_capacity_and_handles_zero_fertile() {
        assert_eq!(calibrate_growth(16384, 0, 100.0), 0);
        let g1 = calibrate_growth(16384, 8192, 100.0);
        let g2 = calibrate_growth(16384, 8192, 200.0);
        assert!(g1 >= 3 && g1 <= 5, "g1 {}", g1); // 100 * 16.384 * 19 / 8192 = 3.8
        assert_eq!(g2, 2 * g1 - (2 * g1 - g2).min(1)); // within rounding of double
    }

    #[test]
    fn terrain_is_periodic_seeded_and_populated() {
        let cfg = WorldConfig { seed: 11, width: 64, height: 64, ..Default::default() };
        let t1 = generate_terrain(&cfg);
        let t2 = generate_terrain(&cfg);
        assert_eq!(t1.grid.cells.len(), 4096);
        for (a, b) in t1.grid.cells.iter().zip(&t2.grid.cells) {
            assert_eq!(a.elevation.to_bits(), b.elevation.to_bits());
            assert_eq!(a.water, b.water);
        }
        let fertile = t1.grid.cells.iter().filter(|c| c.fertile()).count();
        assert!(fertile > 1024 && fertile < 3072, "fertile {} of 4096: expected roughly half", fertile);
        assert_eq!(t1.fertile_cells, fertile);
        for c in &t1.grid.cells {
            assert_eq!(c.get(MAT_SOIL), SOIL0);
            assert!(c.elevation >= 0.0 && c.elevation <= 1.0);
            if c.fertile() {
                assert_eq!(c.get(MAT_AUTOTROPH), AUTOTROPH0);
            }
            if c.bedrock > 0 {
                assert!(c.ore >= 3);
            }
        }
        let volcanic = t1.grid.cells.iter().filter(|c| c.ambient > AMBIENT).count();
        assert!(volcanic > 40 && volcanic < 130, "volcanic {}", volcanic);
        assert!(t1.growth_per_tick > 0);
    }

    #[test]
    fn all_dry_world_still_generates() {
        let cfg = WorldConfig { seed: 5, width: 32, height: 32, ..Default::default() };
        let mut t = generate_terrain(&cfg);
        for c in t.grid.cells.iter_mut() {
            c.water = 0;
        }
        let fertile = t.grid.cells.iter().filter(|c| c.fertile()).count();
        assert_eq!(fertile, 0);
        assert_eq!(calibrate_growth(1024, fertile, 100.0), 0);
    }

    #[test]
    fn harmonics_are_periodic() {
        let mut rng = Rng::new(3);
        let h = Harmonics::new(&mut rng, 64, 32, 6);

        // Test a few sample points for periodicity
        for x in [0.0, 10.5, 23.7, 50.2] {
            for y in [0.0, 5.3, 15.8, 28.9] {
                let v0 = h.at(x, y);
                let vx = h.at(x + 64.0, y);
                let vy = h.at(x, y + 32.0);
                assert!((v0 - vx).abs() < 1e-4, "x period failed at ({}, {})", x, y);
                assert!((v0 - vy).abs() < 1e-4, "y period failed at ({}, {})", x, y);
            }
        }
    }
}
