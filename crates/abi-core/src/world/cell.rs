use crate::chem::generate::MAT_SOIL;
use crate::chem::props::MatId;
use serde::{Deserialize, Serialize};

/// Half of WATER_MAX: the wetness median, so about half the cells are fertile.
pub const FERTILE_WATER: u32 = 1000;

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Cell {
    pub inv: Vec<(MatId, u32)>,
    pub water: u32,
    pub temp: f32,
    pub ambient: f32,
    pub elevation: f32,
    pub bedrock: u32,
    pub ore: MatId,
}

impl Cell {
    pub fn add(&mut self, id: MatId, mass: u32) {
        if mass == 0 {
            return;
        }
        if let Some(e) = self.inv.iter_mut().find(|e| e.0 == id) {
            e.1 += mass;
        } else {
            self.inv.push((id, mass));
        }
    }

    /// Remove up to `mass`; returns what was removed.
    pub fn remove(&mut self, id: MatId, mass: u32) -> u32 {
        if let Some(i) = self.inv.iter().position(|e| e.0 == id) {
            let take = self.inv[i].1.min(mass);
            self.inv[i].1 -= take;
            if self.inv[i].1 == 0 {
                self.inv.swap_remove(i);
            }
            take
        } else {
            0
        }
    }

    pub fn get(&self, id: MatId) -> u32 {
        self.inv.iter().find(|e| e.0 == id).map(|e| e.1).unwrap_or(0)
    }

    pub fn loose_mass(&self) -> u64 {
        self.inv.iter().map(|e| e.1 as u64).sum()
    }

    pub fn mass(&self) -> u64 {
        self.loose_mass() + self.bedrock as u64 + self.water as u64
    }

    pub fn fertile(&self) -> bool {
        self.water >= FERTILE_WATER
    }

    /// Two most massive non-soil items: mass descending, id ascending on ties.
    pub fn top2_non_soil(&self) -> [(MatId, u32); 2] {
        let mut top = [(0 as MatId, 0u32); 2];
        for &(id, m) in &self.inv {
            if id == MAT_SOIL {
                continue;
            }
            if m > top[0].1 || (m == top[0].1 && id < top[0].0) {
                top[1] = top[0];
                top[0] = (id, m);
            } else if m > top[1].1 || (m == top[1].1 && id < top[1].0) {
                top[1] = (id, m);
            }
        }
        top
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_remove_get_are_exact() {
        let mut c = Cell::default();
        c.add(5, 100);
        c.add(5, 50);
        c.add(7, 10);
        assert_eq!(c.get(5), 150);
        assert_eq!(c.remove(5, 200), 150);
        assert_eq!(c.get(5), 0);
        assert_eq!(c.remove(9, 1), 0);
        assert_eq!(c.loose_mass(), 10);
        c.add(3, 0);
        assert_eq!(c.inv.len(), 1, "zero-mass add is a no-op");
    }

    #[test]
    fn mass_includes_bedrock_and_water() {
        let mut c = Cell::default();
        c.add(1, 5);
        c.bedrock = 7;
        c.water = 11;
        assert_eq!(c.mass(), 23);
    }

    #[test]
    fn top2_excludes_soil_and_breaks_ties_by_id() {
        let mut c = Cell::default();
        c.add(0, 9999); // soil
        c.add(4, 100);
        c.add(3, 100);
        c.add(8, 50);
        assert_eq!(c.top2_non_soil(), [(3, 100), (4, 100)]);
        let mut only_soil = Cell::default();
        only_soil.add(0, 100);
        assert_eq!(only_soil.top2_non_soil(), [(0, 0), (0, 0)]);
    }
}
