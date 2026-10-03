use abi_core::chem::generate::ChemParams;
use abi_core::metrics::MetricsRow;
use abi_core::world::config::WorldConfig;
use abi_core::world::World;

fn transitions(seed: u64, ticks: u64) -> f64 {
    let cfg = WorldConfig { seed, width: 128, height: 128, pop0: 1000, graze_capacity_per_1000: 100.0, chem: ChemParams { n_base: 24, k: 2, ..Default::default() }, ..Default::default() };
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

#[test]
#[ignore = "takes ~10 minutes; run with --ignored. The emergence regression: primitives must still produce artifact-based life under scarcity."]
fn reference_seeds_transition_to_artifact_based_life() {
    let mut passed = 0;
    for seed in [7u64, 2, 6] {
        let best = transitions(seed, 60_000);
        eprintln!("seed {} best artifact energy fraction {:.3}", seed, best);
        if best >= 0.5 {
            passed += 1;
        }
    }
    assert!(passed >= 2, "only {} of 3 reference seeds transitioned", passed);
}
