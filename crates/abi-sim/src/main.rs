mod run_dir;

use abi_core::checkpoint;
use abi_core::checkpoint::retention;
use abi_core::chem::generate::ChemParams;
use abi_core::events::{EventLog, ExternalEvent};
use abi_core::hash::state_hash;
use abi_core::metrics::MetricsRow;
use abi_core::world::config::WorldConfig;
use abi_core::world::World;
use clap::{Parser, Subcommand};
use run_dir::RunDir;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "abi-sim", about = "ab initio headless simulation")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct WorldArgs {
    #[arg(long)]
    seed: u64,
    #[arg(long, default_value_t = 256)]
    size: usize,
    #[arg(long, default_value_t = 2000)]
    pop: usize,
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u64).range(1..=256))]
    threads: u64,
    #[arg(long, default_value_t = 100.0)]
    graze: f32,
    #[arg(long, default_value_t = 2)]
    k: usize,
    #[arg(long, default_value_t = 24)]
    base: usize,
}

impl WorldArgs {
    fn config(&self, checkpoint_every: u64) -> WorldConfig {
        WorldConfig { seed: self.seed, width: self.size, height: self.size, pop0: self.pop, graze_capacity_per_1000: self.graze, chem: ChemParams { k: self.k, n_base: self.base, ..Default::default() }, checkpoint_every, ..Default::default() }
    }
}

#[derive(Subcommand)]
enum Cmd {
    Run {
        #[command(flatten)]
        world: WorldArgs,
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
        ticks: u64,
        #[arg(long, default_value_t = 1000, value_parser = clap::value_parser!(u64).range(1..))]
        window: u64,
        #[arg(long, default_value_t = 10_000, value_parser = clap::value_parser!(u64).range(1..))]
        checkpoint_every: u64,
        #[arg(long, default_value_t = 20_000_000_000)]
        disk_budget: u64,
        #[arg(long, default_value = "runs/latest")]
        out: PathBuf,
        #[arg(long)]
        events: Option<PathBuf>,
    },
    Replay {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        to: u64,
        #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u64).range(1..=256))]
        threads: u64,
    },
    Verify {
        #[command(flatten)]
        world: WorldArgs,
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..))]
        ticks: u64,
    },
}

fn pool(n: u64) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(n as usize).build().unwrap()
}

fn parse_events(text: &str) -> Result<EventLog, String> {
    let mut log = EventLog::default();
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let (t, e): (u64, ExternalEvent) = serde_json::from_str(line).map_err(|e| format!("events file line {}: {}", i + 1, e))?;
        if log.entries.last().is_some_and(|last| t < last.0) {
            return Err(format!("events file line {}: tick {} is out of order", i + 1, t));
        }
        log.push(t, e);
    }
    Ok(log)
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run { world, ticks, window, checkpoint_every, disk_budget, out, events } => {
            let cfg = world.config(checkpoint_every);
            let dir = RunDir::create(&out).unwrap_or_else(|e| {
                eprintln!("error: {}: {}", out.display(), e);
                std::process::exit(2);
            });
            dir.write_config(&cfg).expect("write config");
            let mut w = World::new(&cfg);
            if let Some(p) = events {
                let text = std::fs::read_to_string(&p).expect("events file");
                w.events = parse_events(&text).unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    std::process::exit(2);
                });
            }
            dir.write_events(&w.events).expect("write events");
            dir.append_metrics(MetricsRow::header()).expect("write metrics header");
            if let Err(e) = checkpoint::save(&w, &dir.checkpoint_path(0)) {
                eprintln!("warn: checkpoint save failed at tick 0: {}", e);
            }
            let c0 = w.conserved_mass();
            pool(world.threads).install(|| {
                let mut t0 = Instant::now();
                let mut last_row_tick = 0;
                let mut extinct_reported = false;
                while w.tick < ticks {
                    w.step();
                    if w.tick % window == 0 || w.tick == ticks {
                        let secs = t0.elapsed().as_secs_f64();
                        t0 = Instant::now();
                        let stats = std::mem::take(&mut w.stats);
                        let row = MetricsRow::from(&w, &stats, w.tick - last_row_tick, secs);
                        last_row_tick = w.tick;
                        if let Err(e) = dir.append_metrics(&row.to_csv()) {
                            eprintln!("warn: metrics append failed at tick {}: {}", w.tick, e);
                        }
                        assert_eq!(w.conserved_mass(), c0, "mass conservation violated at tick {}", w.tick);
                        eprintln!("tick {} pop {} {:.1} t/s artifacts {:.3}", w.tick, row.pop, row.ticks_per_s, row.artifact_energy_frac);
                    }
                    if w.tick % checkpoint_every == 0 || w.tick == ticks {
                        match dir.list_checkpoints() {
                            Ok(list) => {
                                let ticks_v: Vec<u64> = list.iter().map(|e| e.0).collect();
                                let sizes: Vec<u64> = list.iter().map(|e| e.1).collect();
                                for t in retention::plan(&ticks_v, &sizes, w.tick, disk_budget, checkpoint_every) {
                                    if let Err(e) = std::fs::remove_file(dir.checkpoint_path(t)) {
                                        eprintln!("warn: retention remove of tick {} failed at tick {}: {}", t, w.tick, e);
                                    }
                                }
                            }
                            Err(e) => eprintln!("warn: listing checkpoints failed at tick {}: {}", w.tick, e),
                        }
                        if let Err(e) = checkpoint::save(&w, &dir.checkpoint_path(w.tick)) {
                            eprintln!("warn: checkpoint save failed at tick {}: {}", w.tick, e);
                        }
                    }
                    if w.population() == 0 && !extinct_reported {
                        extinct_reported = true;
                        eprintln!("extinct at tick {}", w.tick);
                    }
                }
            });
            println!("hash {:016x}", state_hash(&w));
        }
        Cmd::Replay { run, to, threads } => {
            let dir = RunDir::open(&run).expect("run dir");
            let list = dir.list_checkpoints().unwrap();
            let (ck_tick, _) = list.iter().rev().find(|e| e.0 <= to).copied().expect("no checkpoint at or before --to");
            let mut w = checkpoint::load(&dir.checkpoint_path(ck_tick)).expect("load checkpoint");
            // The checkpoint carries the event log and its cursor; nothing to re-seek.
            pool(threads).install(|| {
                while w.tick < to {
                    w.step();
                }
            });
            println!("hash {:016x}", state_hash(&w));
        }
        Cmd::Verify { world, ticks } => {
            let cfg = world.config(u64::MAX);
            let run = |n: u64| {
                pool(n).install(|| {
                    let mut w = World::new(&cfg);
                    w.run(ticks);
                    state_hash(&w)
                })
            };
            let (a, b) = (run(1), run(8));
            println!("1 thread : {:016x}\n8 threads: {:016x}\n{}", a, b, if a == b { "DETERMINISTIC" } else { "MISMATCH" });
            std::process::exit(if a == b { 0 } else { 1 });
        }
    }
}
