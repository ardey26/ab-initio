use super::cell::Cell;
use serde::{Deserialize, Serialize};

pub const CHUNK: usize = 32;

#[derive(Clone, Serialize, Deserialize)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Cell>,
}

impl Grid {
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width % CHUNK == 0 && height % CHUNK == 0, "grid dims must be multiples of {}", CHUNK);
        assert!(width <= u16::MAX as usize && height <= u16::MAX as usize);
        Grid { width, height, cells: vec![Cell::default(); width * height] }
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.cells.len()
    }
    /// Chunk-major index: chunk block first, then row-major inside the chunk.
    #[inline]
    pub fn idx(&self, x: u16, y: u16) -> usize {
        let (x, y) = (x as usize, y as usize);
        let chunk = (y / CHUNK) * self.chunks_x() + x / CHUNK;
        chunk * CHUNK * CHUNK + (y % CHUNK) * CHUNK + (x % CHUNK)
    }
    #[inline]
    pub fn xy(&self, idx: usize) -> (u16, u16) {
        let chunk = idx / (CHUNK * CHUNK);
        let local = idx % (CHUNK * CHUNK);
        let cx = chunk % self.chunks_x();
        let cy = chunk / self.chunks_x();
        ((cx * CHUNK + local % CHUNK) as u16, (cy * CHUNK + local / CHUNK) as u16)
    }
    /// dirs: 0 +x, 1 -x, 2 +y, 3 -y (torus)
    #[inline]
    pub fn step(&self, x: u16, y: u16, dir: u8) -> (u16, u16) {
        let (w, h) = (self.width as i32, self.height as i32);
        let (dx, dy) = [(1, 0), (-1, 0), (0, 1), (0, -1)][dir as usize & 3];
        ((((x as i32 + dx) % w + w) % w) as u16, (((y as i32 + dy) % h + h) % h) as u16)
    }
    #[inline]
    pub fn neighbor(&self, idx: usize, dir: u8) -> usize {
        let (x, y) = self.xy(idx);
        let (nx, ny) = self.step(x, y, dir);
        self.idx(nx, ny)
    }
    pub fn chunks_x(&self) -> usize {
        self.width / CHUNK
    }
    pub fn chunks_y(&self) -> usize {
        self.height / CHUNK
    }
    pub fn n_chunks(&self) -> usize {
        self.chunks_x() * self.chunks_y()
    }
    #[inline]
    pub fn chunk_of(&self, idx: usize) -> usize {
        idx / (CHUNK * CHUNK)
    }
    pub fn chunk_cells(&self, chunk: usize) -> std::ops::Range<usize> {
        chunk * CHUNK * CHUNK..(chunk + 1) * CHUNK * CHUNK
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torus_steps_wrap() {
        let g = Grid::new(64, 32);
        assert_eq!(g.step(63, 0, 0), (0, 0));
        assert_eq!(g.step(0, 0, 1), (63, 0));
        assert_eq!(g.step(0, 31, 2), (0, 0));
        assert_eq!(g.step(0, 0, 3), (0, 31));
        assert_eq!(g.neighbor(g.idx(63, 31), 0), g.idx(0, 31));
    }

    #[test]
    fn idx_xy_roundtrip_and_chunk_major() {
        let g = Grid::new(64, 32);
        for y in 0..32u16 {
            for x in 0..64u16 {
                let i = g.idx(x, y);
                assert_eq!(g.xy(i), (x, y));
                assert_eq!(g.chunk_of(i), (y as usize / 32) * 2 + x as usize / 32);
            }
        }
        assert_eq!(g.idx(0, 0), 0);
        assert_eq!(g.idx(32, 0), 1024, "second chunk starts at 1024");
    }

    #[test]
    fn chunks_partition_the_grid() {
        let g = Grid::new(64, 32);
        assert_eq!(g.n_chunks(), 2);
        let mut seen = vec![false; g.len()];
        for ch in 0..g.n_chunks() {
            for c in g.chunk_cells(ch) {
                assert_eq!(g.chunk_of(c), ch);
                assert!(!seen[c]);
                seen[c] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    #[should_panic]
    fn dimensions_must_be_chunk_multiples() {
        let _ = Grid::new(50, 32);
    }
}
