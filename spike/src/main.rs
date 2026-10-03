//! THROWAWAY spike. Subcommands:
//!   chem <n_seeds> [n_base]                       step 1: chemistry richness
//!   evolve <seed> <ticks> [--random] [--threads N] [--size W] [--pop N] [--window W]
//!   determinism <seed> <ticks>                    same seed, 1 vs 8 threads
//!   bench <seed> <ticks> [--pop N] [--size W]     agent-steps/s, memory/agent

mod brain;
mod chem;
mod richness;
mod rng;
mod world;

use std::time::Instant;
use world::*;

fn arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn pool(n: usize) {
    rayon::ThreadPoolBuilder::new().num_threads(n).build_global().ok();
}

fn evolve(args: &[String]) {
    let seed: u64 = args[2].parse().unwrap();
    let ticks: u64 = args[3].parse().unwrap();
    let random = args.iter().any(|a| a == "--random");
    let size: usize = arg(args, "--size", 128);
    let pop: usize = arg(args, "--pop", 2000);
    let window: u64 = arg(args, "--window", 1000);
    let n_base: usize = arg(args, "--base", 24);
    pool(arg(args, "--threads", 8));
    let mut w = World::new(seed, size, size, n_base, pop, 30000, random);
    let m0 = w.total_mass();
    println!("tick,pop,births,deaths,steps_per_s,ms_per_tick,move,take,drop,combine,heat,strike,give,emit,combines_ok,combines_hot,strikes_ok,gives_ok,artifact_energy_frac,artifacts_in_use,held_artifacts,materials_eaten,n_known,mean_energy,env_reactions,env_hot");
    let mut t0 = Instant::now();
    while w.tick < ticks {
        w.step();
        if w.tick % window == 0 {
            let dt = t0.elapsed().as_secs_f64();
            t0 = Instant::now();
            let s = w.take_stats();
            let pop = w.population();
            let steps = s.agent_steps.max(1) as f64;
            let rates: Vec<String> = s.actions.iter().map(|&a| format!("{:.4}", a as f64 / steps)).collect();
            let mean_e: f32 = w.agents.iter().filter(|a| a.alive).map(|a| a.energy).sum::<f32>() / pop.max(1) as f32;
            println!(
                "{},{},{},{},{:.0},{:.2},{},{},{},{},{},{:.4},{},{},{},{},{:.1},{},{}",
                w.tick, pop, s.births, s.deaths, steps / dt, dt * 1000.0 / window as f64, rates.join(","),
                s.combines_ok, s.combines_hot, s.strikes_ok, s.gives_ok,
                if s.energy_total > 0.0 { s.energy_from_artifacts / s.energy_total } else { 0.0 },
                s.artifacts_in_use.len(), w.held_artifacts(), s.materials_eaten.len(), w.chem.n_known(), mean_e, s.env_reactions, s.env_hot
            );
            assert_eq!(w.total_mass(), m0, "mass conservation violated");
            if args.iter().any(|a| a == "--dump") {
                let alive: Vec<&Agent> = w.agents.iter().filter(|a| a.alive).collect();
                let n_plant_cells = w.cells.iter().filter(|c| c.inv.iter().any(|e| e.0 == chem::MAT_PLANT && e.1 >= 1000)).count();
                let can_eat = alive.iter().filter(|a| a.genome.digest_thr < w.chem.props[1][chem::P_NUTRI]).count();
                let (sf, si, pl) = w.soil_split();
                eprintln!("t={} alive={} can_eat_plant={} cells_with_plant>=1000: {}/{} plant_nutri={:.3} soil fertile={}M infertile={}M plant={}M", w.tick, alive.len(), can_eat, n_plant_cells, w.cells.len(), w.chem.props[1][chem::P_NUTRI], sf / 1_000_000, si / 1_000_000, pl / 1_000_000);
                for a in alive.iter().step_by((alive.len() / 6).max(1)).take(6) {
                    let cell = &w.cells[a.y as usize * w.w + a.x as usize];
                    let plant_here = cell.inv.iter().find(|e| e.0 == chem::MAT_PLANT).map(|e| e.1).unwrap_or(0);
                    eprintln!("  id={} age={} E={:.0} body={} thr={:.2} held=[{}:{} n{:.2}, {}:{} n{:.2}] plant_here={} fertile={}",
                        a.id, a.age, a.energy, a.body, a.genome.digest_thr,
                        a.held[0].0, a.held[0].1, w.chem.props[a.held[0].0 as usize][chem::P_NUTRI],
                        a.held[1].0, a.held[1].1, w.chem.props[a.held[1].0 as usize][chem::P_NUTRI], plant_here, cell.fertile);
                }
            }
            if pop == 0 {
                eprintln!("extinct at tick {}", w.tick);
                break;
            }
        }
    }
}

fn determinism(args: &[String]) {
    let seed: u64 = args[2].parse().unwrap();
    let ticks: u64 = args[3].parse().unwrap();
    let run = |threads: usize| {
        let p = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        p.install(|| {
            let mut w = World::new(seed, 64, 64, 16, 1000, 10000, false);
            for _ in 0..ticks {
                w.step();
            }
            (w.state_hash(), w.population(), w.total_mass())
        })
    };
    let a = run(1);
    let b = run(8);
    println!("1 thread : hash {:016x} pop {} mass {}", a.0, a.1, a.2);
    println!("8 threads: hash {:016x} pop {} mass {}", b.0, b.1, b.2);
    println!("{}", if a == b { "DETERMINISTIC" } else { "MISMATCH" });
    std::process::exit(if a == b { 0 } else { 1 });
}

fn bench(args: &[String]) {
    let seed: u64 = args[2].parse().unwrap();
    let ticks: u64 = args[3].parse().unwrap();
    let size: usize = arg(args, "--size", 128);
    let pop: usize = arg(args, "--pop", 8000);
    pool(arg(args, "--threads", 8));
    let mut w = World::new(seed, size, size, 24, pop, 40000, false);
    let t = Instant::now();
    for _ in 0..ticks {
        w.step();
    }
    let dt = t.elapsed().as_secs_f64();
    let s = w.take_stats();
    let agent_bytes = std::mem::size_of::<Agent>() + brain::NW * 4;
    println!("ticks {} in {:.2}s: {:.1} ticks/s, {:.2}M agent-steps/s, {:.2} us/agent-step, pop end {}", ticks, dt, ticks as f64 / dt, s.agent_steps as f64 / dt / 1e6, dt * 1e6 / s.agent_steps as f64, w.population());
    println!("agent struct+genome: {} bytes; {} cells x {} bytes", agent_bytes, w.cells.len(), std::mem::size_of::<Cell>());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("chem") => richness::run(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100), args.get(3).and_then(|s| s.parse().ok()).unwrap_or(24)),
        Some("evolve") => evolve(&args),
        Some("determinism") => determinism(&args),
        Some("bench") => bench(&args),
        Some("pairs") => {
            let seed: u64 = args[2].parse().unwrap();
            let mut c = chem::Chemistry::generate(seed, 24);
            let pn = c.props[1][chem::P_NUTRI];
            println!("seed {} plant nutri {:.3}; plant+X nutrition (cold / hot), hot threshold:", seed, pn);
            for x in 0..24u32 {
                let cold = c.combine(1, x, 0.0);
                let hot = c.combine(1, x, 1.0);
                let th = chem::Chemistry::hot_threshold(&c.props[1], &c.props[x as usize]);
                println!("  X={:2} nutri {:.3} hard {:.3} -> cold {:.3} hot {:.3} (hot needs temp {:.2}){}", x, c.props[x as usize][chem::P_NUTRI], c.props[x as usize][chem::P_HARD],
                    c.props[cold as usize][chem::P_NUTRI], c.props[hot as usize][chem::P_NUTRI], th,
                    if c.props[cold as usize][chem::P_NUTRI] > pn + 0.05 || c.props[hot as usize][chem::P_NUTRI] > pn + 0.05 { "  <-- improves" } else { "" });
            }
        }
        Some("props") => {
            let seed: u64 = args[2].parse().unwrap();
            let c = chem::Chemistry::generate(seed, 24);
            println!("seed {} plant nutri {:.3} toxic {:.3} gain {:.3} | rock hard {:.3} | base max nutri {:.3} max hard {:.3}", seed,
                c.props[1][chem::P_NUTRI], c.props[1][chem::P_TOXIC], c.props[1][chem::P_NUTRI] - 1.5 * c.props[1][chem::P_TOXIC], c.props[2][chem::P_HARD],
                c.props.iter().map(|p| p[chem::P_NUTRI]).fold(0.0, f32::max), c.props.iter().map(|p| p[chem::P_HARD]).fold(0.0, f32::max));
        }
        _ => eprintln!("usage: spike chem|evolve|determinism|bench ..."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mass_is_conserved() {
        let mut w = World::new(7, 32, 32, 12, 300, 5000, false);
        let m0 = w.total_mass();
        for _ in 0..400 {
            w.step();
            assert_eq!(w.total_mass(), m0);
        }
    }

    #[test]
    fn same_seed_same_history_across_thread_counts() {
        let run = |threads: usize| {
            let p = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            p.install(|| {
                let mut w = World::new(3, 32, 32, 12, 300, 5000, false);
                for _ in 0..200 {
                    w.step();
                }
                w.state_hash()
            })
        };
        assert_eq!(run(1), run(4));
    }

    #[test]
    fn combine_is_commutative_and_interned() {
        let mut c = chem::Chemistry::generate(1, 12);
        let x = c.combine(3, 5, 0.0);
        let y = c.combine(5, 3, 0.0);
        assert_eq!(x, y);
        assert_eq!(c.n_known(), 13);
        let z = c.combine(x, 3, 1.0);
        assert!(c.is_artifact(z));
    }
}
