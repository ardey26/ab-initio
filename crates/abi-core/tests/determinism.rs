use abi_core::chem::generate::ChemParams;
use abi_core::hash::state_hash;
use abi_core::world::config::WorldConfig;
use abi_core::world::World;

fn cfg() -> WorldConfig {
    WorldConfig { seed: 21, width: 64, height: 64, pop0: 600, chem: ChemParams { n_base: 16, ..Default::default() }, ..Default::default() }
}

fn run(threads: usize, ticks: u64) -> (u64, usize) {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        let mut w = World::new(&cfg());
        w.run(ticks);
        (state_hash(&w), w.population())
    })
}

#[test]
fn same_seed_same_history_on_1_4_8_threads() {
    let a = run(1, 300);
    let b = run(4, 300);
    let c = run(8, 300);
    assert_eq!(a, b);
    assert_eq!(a, c);
    assert!(a.1 > 0, "population should survive 300 ticks");
}

#[test]
fn large_world_same_history_on_1_and_8_threads() {
    // 128x128 is 16 chunks, so rayon schedules real parallel work across chunks.
    let big = WorldConfig { width: 128, height: 128, pop0: 1500, ..cfg() };
    let run_big = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            let mut w = World::new(&big);
            w.run(150);
            (state_hash(&w), w.population())
        })
    };
    let a = run_big(1);
    let b = run_big(8);
    assert_eq!(a, b);
    assert!(a.1 > 0, "population should survive 150 ticks");
}

#[test]
fn different_seeds_differ() {
    let mut w1 = World::new(&cfg());
    let mut w2 = World::new(&WorldConfig { seed: 22, ..cfg() });
    w1.run(50);
    w2.run(50);
    assert_ne!(state_hash(&w1), state_hash(&w2));
}
