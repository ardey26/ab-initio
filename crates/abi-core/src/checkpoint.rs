//! Full-state checkpoints: bincode + zstd with a small header. Any tick is
//! reachable by loading the nearest earlier checkpoint and stepping forward with
//! the same event log.

use crate::world::World;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"ABI1";
/// 2: `WorldConfig` gained `fire`.
const VERSION: u32 = 2;

pub fn filename(tick: u64) -> String {
    format!("ck-{:012}.bin.zst", tick)
}

/// Writes a checkpoint and returns the bytes written.
///
/// Atomic: data goes to `path` with extension `tmp`, is synced to disk, then
/// renamed onto `path`. A crash mid-write never leaves a truncated file at the
/// final name.
pub fn save(w: &World, path: &Path) -> std::io::Result<u64> {
    let body = bincode::serialize(w).map_err(std::io::Error::other)?;
    let mut out = Vec::with_capacity(body.len() / 3);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    let mut enc = zstd::Encoder::new(&mut out, 3)?;
    enc.write_all(&body)?;
    enc.finish()?;
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(&out)?;
    f.sync_all()?;
    drop(f);
    std::fs::rename(&tmp, path)?;
    Ok(out.len() as u64)
}

/// Loads a checkpoint written by [`save`]. `stats` is empty after load.
///
/// `MaterialTable`'s lookup maps are `#[serde(skip)]` (for deterministic
/// serialization), so this calls `chem.after_load()` to rebuild them. Without
/// it the first `combine` would mint duplicate material ids.
pub fn load(path: &Path) -> std::io::Result<World> {
    let data = std::fs::read(path)?;
    if data.len() < 8 || &data[0..4] != MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not an abi checkpoint"));
    }
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version != VERSION {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("checkpoint version {} != {}", version, VERSION)));
    }
    let mut dec = zstd::Decoder::new(&data[8..])?;
    let mut body = Vec::new();
    dec.read_to_end(&mut body)?;
    let mut w: World = bincode::deserialize(&body).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    w.chem.after_load();
    Ok(w)
}

pub mod retention {
    const RECENT: u64 = 1_000_000;
    const OLD: u64 = 10_000_000;

    fn keep_by_rule(tick: u64, now: u64, every: u64) -> bool {
        if now.saturating_sub(tick) <= RECENT {
            return true;
        }
        let i = tick / every.max(1);
        if now.saturating_sub(tick) <= OLD {
            i % 4 == 0
        } else {
            i % 16 == 0
        }
    }

    /// Returns ticks to delete. `ticks` ascending, `sizes` parallel.
    pub fn plan(ticks: &[u64], sizes: &[u64], now: u64, budget_bytes: u64, every: u64) -> Vec<u64> {
        debug_assert_eq!(ticks.len(), sizes.len());
        if ticks.is_empty() {
            return Vec::new();
        }
        let mut del = Vec::new();
        let mut kept: Vec<(u64, u64)> = Vec::new(); // (tick, size)
        for (i, &t) in ticks.iter().enumerate() {
            if i == 0 || keep_by_rule(t, now, every) {
                kept.push((t, sizes[i]));
            } else {
                del.push(t);
            }
        }
        let mut total: u64 = kept.iter().map(|k| k.1).sum();
        let mut j = 1; // never the earliest, never the newest
        while total > budget_bytes && j + 1 < kept.len() {
            total -= kept[j].1;
            del.push(kept[j].0);
            j += 1;
        }
        del.sort_unstable();
        del
    }
}

#[cfg(test)]
mod tests {
    use super::retention::plan;
    use super::*;

    #[test]
    fn retention_thins_old_checkpoints_and_keeps_first() {
        let every = 10_000u64;
        let ticks: Vec<u64> = (0..1500).map(|i| i * every).collect(); // up to 15M
        let sizes = vec![1u64; ticks.len()];
        let now = 15_000_000;
        let del = plan(&ticks, &sizes, now, u64::MAX, every);
        assert!(!del.contains(&0));
        assert!(!del.contains(&14_990_000), "recent kept");
        assert!(!del.contains(&14_010_000), "within 1M of now kept");
        assert!(del.contains(&13_990_000), "1.01M ago, index 1399 % 4 != 0 -> deleted");
        assert!(!del.contains(&13_960_000), "index 1396 % 4 == 0 -> kept");
        assert!(del.contains(&4_040_000), "older than 10M: index 404 % 16 != 0 -> deleted");
        assert!(!del.contains(&4_000_000), "index 400 % 16 == 0 -> kept");
    }

    #[test]
    fn retention_enforces_budget_oldest_first() {
        let every = 10u64;
        let ticks: Vec<u64> = (0..100).map(|i| i * every).collect();
        let sizes = vec![10u64; 100];
        let del = plan(&ticks, &sizes, 990, 500, every); // budget allows 50 of 100
        assert_eq!(del.len(), 50);
        assert!(!del.contains(&0));
        assert!(del.contains(&10), "oldest deletable goes first");
        assert!(!del.contains(&990));
    }

    #[test]
    fn budget_never_deletes_newest() {
        assert!(plan(&[0, 10], &[10, 10], 10, 5, 10).is_empty());
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("abi-ck-unit-{}-{}.bin.zst", std::process::id(), name))
    }

    fn small_world() -> World {
        use crate::chem::generate::ChemParams;
        use crate::world::config::WorldConfig;
        World::new(&WorldConfig { seed: 5, width: 32, height: 32, pop0: 10, chem: ChemParams { n_base: 8, ..Default::default() }, ..Default::default() })
    }

    #[test]
    fn load_rejects_bad_magic() {
        let p = tmp_path("magic");
        let mut data = b"NOPE".to_vec();
        data.extend_from_slice(&VERSION.to_le_bytes());
        std::fs::write(&p, &data).unwrap();
        let err = load(&p).err().expect("bad magic must fail");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn load_rejects_wrong_version() {
        let p = tmp_path("version");
        save(&small_world(), &p).unwrap();
        let mut data = std::fs::read(&p).unwrap();
        data[4..8].copy_from_slice(&(VERSION + 1).to_le_bytes());
        std::fs::write(&p, &data).unwrap();
        let err = load(&p).err().expect("wrong version must fail");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn load_rejects_truncated_file() {
        let p = tmp_path("trunc");
        save(&small_world(), &p).unwrap();
        let data = std::fs::read(&p).unwrap();
        std::fs::write(&p, &data[..data.len() / 2]).unwrap();
        assert!(load(&p).is_err());
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn save_leaves_no_tmp_file() {
        let p = tmp_path("atomic");
        save(&small_world(), &p).unwrap();
        assert!(p.exists());
        assert!(!p.with_extension("tmp").exists());
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn filename_sorts_lexicographically_by_tick() {
        assert!(filename(9) < filename(10));
        assert!(filename(999_999) < filename(1_000_000));
    }
}
