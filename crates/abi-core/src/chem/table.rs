//! Material table. A material is its quantized squashed property vector. Ids are
//! never reassigned. GC drops the data of ids that no longer exist anywhere; the
//! bin-to-id map is kept so the bin gets its old id back when produced again.

use super::props::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct Recipe {
    pub a: MatId,
    pub b: MatId,
    pub tq: u8,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct MaterialTable {
    n_base: usize,
    raw: Vec<Props>,
    props: Vec<Props>,
    recipe: Vec<Option<Recipe>>,
    evicted: Vec<bool>,
    by_bin: HashMap<BinKey, MatId>,
    by_recipe: HashMap<Recipe, MatId>,
}

impl MaterialTable {
    pub fn new(base_raw: Vec<Props>) -> Self {
        let n_base = base_raw.len();
        let mut t = MaterialTable { n_base, raw: Vec::new(), props: Vec::new(), recipe: Vec::new(), evicted: Vec::new(), by_bin: HashMap::new(), by_recipe: HashMap::new() };
        for r in base_raw {
            let r = clamp_raw(&r);
            let p = squash_all(&r);
            let id = t.raw.len() as MatId;
            t.by_bin.entry(quantize(&p)).or_insert(id);
            t.raw.push(r);
            t.props.push(p);
            t.recipe.push(None);
            t.evicted.push(false);
        }
        t
    }

    pub fn len(&self) -> usize {
        self.raw.len()
    }
    pub fn n_base(&self) -> usize {
        self.n_base
    }
    pub fn is_artifact(&self, id: MatId) -> bool {
        id as usize >= self.n_base
    }
    pub fn is_evicted(&self, id: MatId) -> bool {
        self.evicted[id as usize]
    }
    pub fn props(&self, id: MatId) -> &Props {
        debug_assert!(!self.evicted[id as usize], "material {} evicted", id);
        &self.props[id as usize]
    }
    pub fn raw(&self, id: MatId) -> &Props {
        &self.raw[id as usize]
    }
    pub fn recipe(&self, id: MatId) -> Option<Recipe> {
        self.recipe[id as usize]
    }
    pub fn cached(&self, recipe: &Recipe) -> Option<MatId> {
        self.by_recipe.get(recipe).copied()
    }

    /// Intern a reaction result. Returns the id of its bin, reviving an evicted id
    /// if the bin was seen before.
    pub fn intern(&mut self, raw: Props, recipe: Recipe) -> MatId {
        let raw = clamp_raw(&raw);
        let p = squash_all(&raw);
        let key = quantize(&p);
        let id = match self.by_bin.get(&key) {
            Some(&id) => {
                if self.evicted[id as usize] {
                    self.raw[id as usize] = raw;
                    self.props[id as usize] = p;
                    self.recipe[id as usize] = Some(recipe);
                    self.evicted[id as usize] = false;
                }
                id
            }
            None => {
                let id = self.raw.len() as MatId;
                self.raw.push(raw);
                self.props.push(p);
                self.recipe.push(Some(recipe));
                self.evicted.push(false);
                self.by_bin.insert(key, id);
                id
            }
        };
        self.by_recipe.insert(recipe, id);
        id
    }

    /// `live[id]` says whether any mass of `id` exists or any agent holds it.
    /// Evicts non-base dead ids: drops recipe cache entries pointing at them.
    pub fn gc(&mut self, live: &[bool]) -> usize {
        let mut n = 0;
        for id in self.n_base..self.raw.len() {
            if !self.evicted[id] && !live.get(id).copied().unwrap_or(false) {
                self.evicted[id] = true;
                n += 1;
            }
        }
        if n > 0 {
            let evicted = &self.evicted;
            self.by_recipe.retain(|_, &mut id| !evicted[id as usize]);
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chem::generate::{generate_base, ChemParams};
    use crate::chem::Chemistry;

    fn raw_with_nutri(n: f32) -> Props {
        let mut p = [0f32; NP];
        p[P_NUTRI] = logit(n);
        p
    }

    #[test]
    fn same_bin_same_id_different_bin_new_id() {
        let mut t = MaterialTable::new(generate_base(1, 4));
        let r1 = Recipe { a: 1, b: 2, tq: 0 };
        let r2 = Recipe { a: 1, b: 3, tq: 0 };
        let id1 = t.intern(raw_with_nutri(0.80), r1);
        let id2 = t.intern(raw_with_nutri(0.82), r2); // same bin (0.8 and 0.82 both round to level 3)
        assert_eq!(id1, id2);
        assert_eq!(t.recipe(id1), Some(r1), "first recipe is kept");
        let id3 = t.intern(raw_with_nutri(0.95), r2);
        assert_ne!(id3, id1);
        assert!(t.is_artifact(id3));
        assert_eq!(t.len(), 6);
    }

    #[test]
    fn gc_evicts_dead_ids_and_revives_them_with_same_id() {
        let mut t = MaterialTable::new(generate_base(1, 4));
        let r = Recipe { a: 1, b: 2, tq: 0 };
        let id = t.intern(raw_with_nutri(0.80), r);
        let props_before = *t.props(id);
        let mut live = vec![true; t.len()];
        live[id as usize] = false;
        assert_eq!(t.gc(&live), 1);
        assert!(t.is_evicted(id));
        let again = t.intern(raw_with_nutri(0.80), r);
        assert_eq!(again, id);
        assert!(!t.is_evicted(id));
        assert_eq!(*t.props(id), props_before);
    }

    #[test]
    fn base_materials_are_never_evicted() {
        let mut t = MaterialTable::new(generate_base(1, 4));
        let live = vec![false; 4];
        assert_eq!(t.gc(&live), 0);
    }

    #[test]
    fn combine_is_commutative_and_cached() {
        let mut c = Chemistry::new(1, &ChemParams::default());
        let x = c.combine(3, 5, 0.2);
        let y = c.combine(5, 3, 0.2);
        assert_eq!(x, y);
        let before = c.table.len();
        let _ = c.combine(3, 5, 0.2);
        assert_eq!(c.table.len(), before);
        let z = c.combine(x, 3, 0.9);
        assert!(z < c.table.len() as MatId);
    }

    #[test]
    fn temperature_buckets() {
        assert_eq!(Chemistry::temp_bucket(0.0), 0);
        assert_eq!(Chemistry::temp_bucket(0.2), 0);
        assert_eq!(Chemistry::temp_bucket(0.5), 1);
        assert_eq!(Chemistry::temp_bucket(0.9), 2);
        assert_eq!(Chemistry::temp_bucket(5.0), 3);
        assert!(!Chemistry::hot(0.3));
        assert!(Chemistry::hot(0.31));
    }
}
