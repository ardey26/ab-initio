//! Exact, deterministic, parallel diffusion. Pass 1 computes each cell's outflow
//! to its +x and +y neighbours from the frozen state. Pass 2 lets each cell apply
//! its own outflows and the inflows from its -x and -y neighbours. Each cell is
//! written by exactly one task, so thread count cannot change the result.

use super::cell::Cell;
use super::grid::Grid;
use crate::chem::generate::MAT_SOIL;
use rayon::prelude::*;

/// Signed flux from cell c to its +x neighbour (`.0`) and +y neighbour (`.1`).
/// Positive means c loses mass.
fn fluxes(grid: &Grid, get: &(dyn Fn(&Cell) -> u32 + Sync), divisor: u32) -> Vec<(i64, i64)> {
    (0..grid.len())
        .into_par_iter()
        .map(|c| {
            let v = get(&grid.cells[c]) as i64;
            let mut out = (0i64, 0i64);
            for (k, dir) in [0u8, 2u8].iter().enumerate() {
                let n = grid.neighbor(c, *dir);
                let vn = get(&grid.cells[n]) as i64;
                let f = (v - vn) / divisor as i64; // truncation toward zero keeps |f| <= |v - vn| / divisor
                if k == 0 {
                    out.0 = f;
                } else {
                    out.1 = f;
                }
            }
            out
        })
        .collect()
}

pub fn diffuse_u32(grid: &mut Grid, get: &(dyn Fn(&Cell) -> u32 + Sync), set: &(dyn Fn(&mut Cell, u32) + Sync), divisor: u32) {
    let fl = fluxes(grid, get, divisor);
    let g2 = Grid { width: grid.width, height: grid.height, cells: Vec::new() }; // index helper only
    grid.cells.par_iter_mut().enumerate().for_each(|(c, cell)| {
        let v = get(cell) as i64;
        let left = g2.neighbor(c, 1);
        let up = g2.neighbor(c, 3);
        let nv = v - fl[c].0 - fl[c].1 + fl[left].0 + fl[up].1;
        debug_assert!(nv >= 0, "negative mass after diffusion at {}", c);
        set(cell, nv.max(0) as u32);
    });
}

pub fn diffuse_soil(grid: &mut Grid) {
    diffuse_u32(
        grid,
        &|c: &Cell| c.get(MAT_SOIL),
        &|c: &mut Cell, v: u32| {
            let cur = c.get(MAT_SOIL);
            if v > cur {
                c.add(MAT_SOIL, v - cur);
            } else {
                c.remove(MAT_SOIL, cur - v);
            }
        },
        8,
    );
}

pub fn diffuse_water(grid: &mut Grid) {
    diffuse_u32(grid, &|c: &Cell| c.water, &|c: &mut Cell, v: u32| c.water = v, 8);
}

/// Decay toward ambient, then average with 4 neighbours (weight 0.1 each).
pub fn relax_temperature(grid: &mut Grid, decay: f32) {
    let frozen: Vec<f32> = grid.cells.iter().map(|c| c.temp).collect();
    let g = &*grid;
    let next: Vec<f32> = (0..g.len())
        .into_par_iter()
        .map(|c| {
            let mut s = 0f32;
            for dir in 0..4u8 {
                s += frozen[g.neighbor(c, dir)];
            }
            let mixed = frozen[c] * 0.6 + s * 0.1;
            let amb = g.cells[c].ambient;
            amb + (mixed - amb) * decay
        })
        .collect();
    grid.cells.par_iter_mut().zip(next.par_iter()).for_each(|(c, &t)| c.temp = t);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total_soil(g: &Grid) -> u64 {
        g.cells.iter().map(|c| c.get(MAT_SOIL) as u64).sum()
    }

    #[test]
    fn soil_diffusion_conserves_and_flattens() {
        let mut g = Grid::new(32, 32);
        g.cells[0].add(MAT_SOIL, 1_000_000);
        let before = total_soil(&g);
        for _ in 0..200 {
            diffuse_soil(&mut g);
            assert_eq!(total_soil(&g), before);
        }
        let max = g.cells.iter().map(|c| c.get(MAT_SOIL)).max().unwrap();
        let min = g.cells.iter().map(|c| c.get(MAT_SOIL)).min().unwrap();
        assert!(max - min < 200_000, "spread: max {} min {}", max, min);
    }

    #[test]
    fn water_diffusion_conserves() {
        let mut g = Grid::new(32, 32);
        for (i, c) in g.cells.iter_mut().enumerate() {
            c.water = (i as u32 * 7919) % 5000;
        }
        let before: u64 = g.cells.iter().map(|c| c.water as u64).sum();
        for _ in 0..50 {
            diffuse_water(&mut g);
        }
        let after: u64 = g.cells.iter().map(|c| c.water as u64).sum();
        assert_eq!(before, after);
    }

    #[test]
    fn diffusion_is_thread_count_independent() {
        let make = || {
            let mut g = Grid::new(64, 64);
            for (i, c) in g.cells.iter_mut().enumerate() {
                c.add(MAT_SOIL, (i as u32).wrapping_mul(2654435761u32) % 9000);
                c.water = (i as u32).wrapping_mul(40503) % 3000;
                c.temp = 0.2 + ((i % 13) as f32) * 0.05;
                c.ambient = 0.2;
            }
            g
        };
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut g = make();
                for _ in 0..30 {
                    diffuse_soil(&mut g);
                    diffuse_water(&mut g);
                    relax_temperature(&mut g, 0.85);
                }
                g.cells.iter().map(|c| (c.get(MAT_SOIL), c.water, c.temp.to_bits())).collect::<Vec<_>>()
            })
        };
        assert_eq!(run(1), run(8));
    }

    #[test]
    fn temperature_relaxes_to_ambient() {
        let mut g = Grid::new(32, 32);
        for c in g.cells.iter_mut() {
            c.ambient = 0.2;
            c.temp = 0.2;
        }
        g.cells[100].temp = 1.0;
        for _ in 0..100 {
            relax_temperature(&mut g, 0.85);
        }
        assert!((g.cells[100].temp - 0.2).abs() < 1e-3);
    }
}
