//! THROWAWAY spike. Subcommands:
//!   chem <n_seeds> [--base N] [--k K] [--amp A] [--rho R]     chemistry richness
//!   evolve <seed> <ticks> [--random] [--nolearn] [--threads N] [--size W] [--pop N] [--window W] [--k K] [--amp A] [--rho R] [--dump]
//!   determinism <seed> <ticks>
//!   bench <seed> <ticks> [--pop N] [--size W]
//!   pairs <seed> [--k K]       plant+X outcomes

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

fn config(args: &[String], seed: u64) -> Config {
    Config {
        seed,
        w: arg(args, "--size", 128),
        h: arg(args, "--size", 128),
        n_base: arg(args, "--base", 24),
        pop0: arg(args, "--pop", 1000),
        max_agents: 40000,
        random_policy: args.iter().any(|a| a == "--random"),
        learn: !args.iter().any(|a| a == "--nolearn"),
        k: arg(args, "--k", 2),
        amplitude: arg(args, "--amp", 1.0),
        rho: arg(args, "--rho", 1.0),
        growth: arg(args, "--growth", 40),
    }
}

const HEADER: &str = "tick,pop,births,deaths,steps_per_s,ms_per_tick,move,take,drop,combine,heat,strike,give,emit,combines_ok,combines_hot,strikes_ok,gives_ok,artifact_energy_frac,artifacts_in_use,held_artifacts,materials_eaten,n_known,mean_energy,env_reactions,env_hot,mean_eta,mean_imit";

fn evolve(args: &[String]) {
    let seed: u64 = args[2].parse().unwrap();
    let ticks: u64 = args[3].parse().unwrap();
    let window: u64 = arg(args, "--window", 1000);
    pool(arg(args, "--threads", 8));
    let cfg = config(args, seed);
    let mut w = World::new(&cfg);
    let m0 = w.total_mass();
    println!("{}", HEADER);
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
            let (eta, imit) = w.mean_learning();
            println!(
                "{},{},{},{},{:.0},{:.2},{},{},{},{},{},{:.4},{},{},{},{},{:.1},{},{},{:.5},{:.5}",
                w.tick, pop, s.births, s.deaths, steps / dt, dt * 1000.0 / window as f64, rates.join(","),
                s.combines_ok, s.combines_hot, s.strikes_ok, s.gives_ok,
                if s.energy_total > 0.0 { s.energy_from_artifacts / s.energy_total } else { 0.0 },
                s.artifacts_in_use.len(), w.held_artifacts(), s.materials_eaten.len(), w.chem.n_known(), mean_e, s.env_reactions, s.env_hot, eta, imit
            );
            assert_eq!(w.total_mass(), m0, "mass conservation violated");
            if args.iter().any(|a| a == "--dump") {
                let alive: Vec<&Agent> = w.agents.iter().filter(|a| a.alive).collect();
                let (sf, si, pl) = w.soil_split();
                eprintln!("t={} alive={} soil fertile={}M infertile={}M plant={}M", w.tick, alive.len(), sf / 1_000_000, si / 1_000_000, pl / 1_000_000);
                for a in alive.iter().step_by((alive.len() / 4).max(1)).take(4) {
                    eprintln!("  id={} age={} E={:.0} body={} thr={:.2} eta={:.4} imit={:.4} held=[{}:{} n{:.2}, {}:{} n{:.2}]",
                        a.id, a.age, a.energy, a.body, a.genome.digest_thr, a.genome.eta, a.genome.imit,
                        a.held[0].0, a.held[0].1, w.chem.props[a.held[0].0 as usize][chem::P_NUTRI],
                        a.held[1].0, a.held[1].1, w.chem.props[a.held[1].0 as usize][chem::P_NUTRI]);
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
            let mut cfg = config(args, seed);
            cfg.w = 64;
            cfg.h = 64;
            cfg.n_base = 16;
            let mut w = World::new(&cfg);
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
    pool(arg(args, "--threads", 8));
    let mut cfg = config(args, seed);
    cfg.pop0 = arg(args, "--pop", 8000);
    let mut w = World::new(&cfg);
    let t = Instant::now();
    for _ in 0..ticks {
        w.step();
    }
    let dt = t.elapsed().as_secs_f64();
    let s = w.take_stats();
    let agent_bytes = std::mem::size_of::<Agent>() + 2 * brain::NW * 4 + brain::NH * brain::NACT * 4;
    println!("ticks {} in {:.2}s: {:.1} ticks/s, {:.2}M agent-steps/s, {:.2} us/agent-step, pop end {}", ticks, dt, ticks as f64 / dt, s.agent_steps as f64 / dt / 1e6, dt * 1e6 / s.agent_steps as f64, w.population());
    println!("agent struct+genome+lifetime weights+trace: {} bytes; {} cells x {} bytes", agent_bytes, w.cells.len(), std::mem::size_of::<Cell>());
}

fn pairs(args: &[String]) {
    let seed: u64 = args[2].parse().unwrap();
    let cfg = config(args, seed);
    let mut c = chem::Chemistry::generate(seed, cfg.n_base, cfg.k, cfg.amplitude, cfg.rho);
    let pn = c.props[1][chem::P_NUTRI];
    println!("seed {} K={} plant nutri {:.3}; plant+X nutrition at temp 0.2 / 0.5 / 0.9:", seed, cfg.k, pn);
    for x in 0..cfg.n_base as u32 {
        let r: Vec<f32> = [0.2f32, 0.5, 0.9].iter().map(|&t| { let id = c.combine(1, x, t); c.props[id as usize][chem::P_NUTRI] }).collect();
        let best = r.iter().cloned().fold(0.0, f32::max);
        println!("  X={:2} nutri {:.3} hard {:.3} -> {:.3} {:.3} {:.3}{}", x, c.props[x as usize][chem::P_NUTRI], c.props[x as usize][chem::P_HARD], r[0], r[1], r[2], if best > pn + 0.05 { "  <-- improves" } else { "" });
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("chem") => richness::run(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100), arg(&args, "--base", 24), arg(&args, "--k", 2), arg(&args, "--amp", 1.0), arg(&args, "--rho", 1.0)),
        Some("evolve") => evolve(&args),
        Some("determinism") => determinism(&args),
        Some("bench") => bench(&args),
        Some("pairs") => pairs(&args),
        _ => eprintln!("usage: spike chem|evolve|determinism|bench|pairs ..."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small(seed: u64) -> Config {
        Config { seed, w: 32, h: 32, n_base: 12, pop0: 300, max_agents: 5000, random_policy: false, learn: true, k: 2, amplitude: 1.0, rho: 1.0, growth: 40 }
    }

    #[test]
    fn mass_is_conserved() {
        let mut w = World::new(&small(7));
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
                let mut w = World::new(&small(3));
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
        let mut c = chem::Chemistry::generate(1, 12, 2, 1.0, 1.0);
        let x = c.combine(3, 5, 0.2);
        let y = c.combine(5, 3, 0.2);
        assert_eq!(x, y);
        let z = c.combine(x, 3, 0.9);
        assert!(c.is_artifact(z) || z < 12);
    }

    #[test]
    fn k_zero_outputs_depend_only_on_own_property() {
        let c = chem::Chemistry::generate(5, 12, 0, 1.0, 1.0);
        let a = c.raw[3];
        let mut b = c.raw[4];
        let r1 = c.react(&a, &b, 0.2);
        b[chem::P_HARD] += 1.0; // change one input property
        let r2 = c.react(&a, &b, 0.2);
        for j in 0..chem::NP {
            if j != chem::P_HARD {
                assert!((r1[j] - r2[j]).abs() < 1e-6, "property {} changed under K=0", j);
            }
        }
    }
}
