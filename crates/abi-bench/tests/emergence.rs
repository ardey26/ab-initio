use abi_core::chem::generate::ChemParams;
use abi_core::metrics::MetricsRow;
use abi_core::world::config::WorldConfig;
use abi_core::world::World;

fn transitions(seed: u64, ticks: u64) -> f64 {
    // fire is explicitly off: the reference seeds were selected with fire off.
    let cfg = WorldConfig { seed, width: 128, height: 128, pop0: 1000, graze_capacity_per_1000: 100.0, chem: ChemParams { n_base: 24, k: 2, ..Default::default() }, fire: false, ..Default::default() };
    let mut w = World::new(&cfg);
    let mut best = 0.0f64;
    let window = 4000;
    while w.tick < ticks {
        w.run(window);
        let stats = std::mem::take(&mut w.stats);
        let row = MetricsRow::from(&w, &stats, window, 1.0);
        eprintln!("seed {} tick {} pop {} artifacts {:.3} combine {:.3}", seed, w.tick, row.pop, row.artifact_energy_frac, row.action_rates[3]);
        best = best.max(row.artifact_energy_frac);
        if row.pop == 0 {
            break;
        }
    }
    best
}

/// Fixed-seed canary, not a statistic; a change that flips seeds must re-select
/// with justification in the commit. Seeds are the 4 of 21 that passed in the
/// Task 19 scan (seeds 0..20, fire off), best fraction: 6 (0.916),
/// 0 (0.836), 20 (0.692), 18 (0.646).
#[test]
#[ignore = "takes ~11 minutes; run with --ignored. The emergence regression: primitives must still produce artifact-based life under scarcity."]
fn reference_seeds_transition_to_artifact_based_life() {
    let mut passed = 0;
    for seed in [6u64, 0, 20, 18] {
        let best = transitions(seed, 60_000);
        eprintln!("seed {} best artifact energy fraction {:.3}", seed, best);
        if best >= 0.5 {
            passed += 1;
        }
    }
    assert!(passed >= 2, "only {} of 4 reference seeds transitioned", passed);
}
