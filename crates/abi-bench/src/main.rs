use abi_core::checkpoint;
use abi_core::chem::generate::ChemParams;
use abi_core::hash::state_hash;
use abi_core::metrics::MetricsRow;
use abi_core::world::config::WorldConfig;
use abi_core::world::step::PhaseTimes;
use abi_core::world::World;
use clap::{Parser, Subcommand};
use std::time::Instant;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct Shape {
    #[arg(long, default_value_t = 256)]
    size: usize,
    /// Founders. Default: 1.5x the grazing capacity of the default shape
    /// (100 per 1000 cells x 65,536 cells = 6554), so the world does not crash.
    #[arg(long, default_value_t = 9800)]
    pop: usize,
    /// Default: the best surviving seed of the Task 19 scan (0.916 artifact energy).
    #[arg(long, default_value_t = 6)]
    seed: u64,
    #[arg(long, default_value_t = 100.0)]
    graze: f32,
}

impl Shape {
    fn cfg(&self) -> WorldConfig {
        // fire is explicitly off: the reference seeds were selected with fire off.
        WorldConfig { seed: self.seed, width: self.size, height: self.size, pop0: self.pop, graze_capacity_per_1000: self.graze, chem: ChemParams::default(), fire: false, ..Default::default() }
    }
}

// Shape for the emergence run: its own defaults (128x128, 1000 founders), the
// size and founder count the reference seeds were scanned at.
#[derive(clap::Args, Clone)]
struct EmergenceShape {
    #[arg(long, default_value_t = 128)]
    size: usize,
    #[arg(long, default_value_t = 1000)]
    pop: usize,
    #[arg(long, default_value_t = 6)]
    seed: u64,
    #[arg(long, default_value_t = 100.0)]
    graze: f32,
}

impl EmergenceShape {
    fn cfg(&self) -> WorldConfig {
        Shape { size: self.size, pop: self.pop, seed: self.seed, graze: self.graze }.cfg()
    }
}

#[derive(Subcommand)]
enum Cmd {
    Throughput { #[command(flatten)] shape: Shape, #[arg(long, default_value_t = 500)] ticks: u64, #[arg(long, default_value_t = 8)] threads: usize, #[arg(long, default_value_t = 200)] warmup: u64 },
    /// Per-phase share of the tick.
    Phases { #[command(flatten)] shape: Shape, #[arg(long, default_value_t = 300)] ticks: u64, #[arg(long, default_value_t = 8)] threads: usize, #[arg(long, default_value_t = 100)] warmup: u64 },
    /// Memory per agent; `--warmup` runs the world first so an evolved population can be measured.
    Memory { #[command(flatten)] shape: Shape, #[arg(long, default_value_t = 0)] warmup: u64 },
    Speedup { #[command(flatten)] shape: Shape, #[arg(long, default_value_t = 300)] ticks: u64 },
    Emergence { #[command(flatten)] shape: EmergenceShape, #[arg(long, default_value_t = 60000)] ticks: u64, #[arg(long, default_value_t = 8)] threads: usize },
}

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap()
}

fn throughput(cfg: &WorldConfig, ticks: u64, warmup: u64, threads: usize) -> (f64, f64, f64, usize) {
    pool(threads).install(|| {
        let mut w = World::new(cfg);
        w.run(warmup);
        w.stats = Default::default();
        let t = Instant::now();
        w.run(ticks);
        let secs = t.elapsed().as_secs_f64();
        let steps = w.stats.agent_steps as f64;
        (ticks as f64 / secs, steps / secs, secs * 1e6 / steps.max(1.0), w.population())
    })
}

fn main() {
    match Cli::parse().cmd {
        Cmd::Throughput { shape, ticks, threads, warmup } => {
            let (tps, sps, us, pop) = throughput(&shape.cfg(), ticks, warmup, threads);
            println!("ticks_per_s {:.1}\nagent_steps_per_s {:.0}\nus_per_agent_step {:.3}\npop_end {}", tps, sps, us, pop);
        }
        Cmd::Phases { shape, ticks, threads, warmup } => {
            let mut t = PhaseTimes::default();
            let pop = pool(threads).install(|| {
                let mut w = World::new(&shape.cfg());
                w.run(warmup);
                for _ in 0..ticks {
                    w.step_timed(&mut t);
                }
                w.population()
            });
            let total = t.total().as_secs_f64();
            for (name, d) in t.rows() {
                println!("{:<18} {:>9.3} ms/tick {:>6.1}%", name, d.as_secs_f64() * 1e3 / ticks as f64, 100.0 * d.as_secs_f64() / total);
            }
            println!("{:<18} {:>9.3} ms/tick\npop_end {}", "total", total * 1e3 / ticks as f64, pop);
        }
        Cmd::Memory { shape, warmup } => {
            let mut w = World::new(&shape.cfg());
            pool(8).install(|| w.run(warmup));
            let n = w.agents.len().max(1);
            let bytes: usize = w.agents.iter().map(|a| std::mem::size_of::<abi_core::agent::Agent>() + a.genome.bytes()).sum();
            let path = std::env::temp_dir().join("abi-bench-ck.bin.zst");
            let ck = checkpoint::save(&w, &path).unwrap();
            let _ = std::fs::remove_file(&path);
            println!("bytes_per_agent {}\ncheckpoint_bytes {}\nmaterials {}", bytes / n, ck, w.chem.table.len());
        }
        Cmd::Speedup { shape, ticks } => {
            let cfg = shape.cfg();
            let (t1, ..) = throughput(&cfg, ticks, 50, 1);
            let (t8, ..) = throughput(&cfg, ticks, 50, 8);
            let h1 = pool(1).install(|| { let mut w = World::new(&cfg); w.run(100); state_hash(&w) });
            let h8 = pool(8).install(|| { let mut w = World::new(&cfg); w.run(100); state_hash(&w) });
            println!("ticks_per_s_1 {:.1}\nticks_per_s_8 {:.1}\nspeedup {:.2}\ndeterministic {}", t1, t8, t8 / t1, h1 == h8);
        }
        Cmd::Emergence { shape, ticks, threads } => {
            let cfg = shape.cfg();
            pool(threads).install(|| {
                let mut w = World::new(&cfg);
                let mut best = 0.0f64;
                let window = 4000;
                println!("{}", MetricsRow::header());
                while w.tick < ticks {
                    let t = Instant::now();
                    w.run(window);
                    let stats = std::mem::take(&mut w.stats);
                    let row = MetricsRow::from(&w, &stats, window, t.elapsed().as_secs_f64());
                    println!("{}", row.to_csv());
                    best = best.max(row.artifact_energy_frac);
                    if row.pop == 0 { break; }
                }
                println!("{} best_artifact_energy_frac {:.3}", if best >= 0.5 { "PASS" } else { "FAIL" }, best);
            });
        }
    }
}
