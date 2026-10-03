use abi_core::checkpoint;
use abi_core::events::EventLog;
use abi_core::world::config::WorldConfig;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct RunDir {
    pub root: PathBuf,
}

impl RunDir {
    pub fn create(root: &Path) -> std::io::Result<RunDir> {
        fs::create_dir_all(root.join("checkpoints"))?;
        Ok(RunDir { root: root.to_path_buf() })
    }
    pub fn open(root: &Path) -> std::io::Result<RunDir> {
        if !root.join("checkpoints").is_dir() {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "not a run directory"));
        }
        Ok(RunDir { root: root.to_path_buf() })
    }
    pub fn write_config(&self, cfg: &WorldConfig) -> std::io::Result<()> {
        let s = toml::to_string_pretty(cfg).map_err(std::io::Error::other)?;
        fs::write(self.root.join("config.toml"), s)
    }
    pub fn append_metrics(&self, line: &str) -> std::io::Result<()> {
        let mut f = fs::OpenOptions::new().create(true).append(true).open(self.root.join("metrics.csv"))?;
        writeln!(f, "{}", line)
    }
    pub fn checkpoint_path(&self, tick: u64) -> PathBuf {
        self.root.join("checkpoints").join(checkpoint::filename(tick))
    }
    /// (tick, size) ascending by tick.
    pub fn list_checkpoints(&self) -> std::io::Result<Vec<(u64, u64)>> {
        let mut v = Vec::new();
        for e in fs::read_dir(self.root.join("checkpoints"))? {
            let e = e?;
            let name = e.file_name().to_string_lossy().to_string();
            if let Some(t) = name.strip_prefix("ck-").and_then(|s| s.strip_suffix(".bin.zst")).and_then(|s| s.parse::<u64>().ok()) {
                v.push((t, e.metadata()?.len()));
            }
        }
        v.sort_unstable();
        Ok(v)
    }
    pub fn write_events(&self, log: &EventLog) -> std::io::Result<()> {
        let mut f = fs::File::create(self.root.join("events.log"))?;
        for (t, e) in &log.entries {
            writeln!(f, "{}", serde_json::to_string(&(t, e))?)?;
        }
        Ok(())
    }
}
