use abi_core::chem::generate::ChemParams;
use abi_core::events::ExternalEvent;
use abi_core::hash::state_hash;
use abi_core::world::config::WorldConfig;
use abi_core::world::World;

/// Changing this value means history changed. If that is intended, update it in the same commit and say why.
const GOLDEN: u64 = 0x219904eadf4bab05;

fn run(threads: usize) -> u64 {
    let cfg = WorldConfig { seed: 1234, width: 64, height: 64, pop0: 400, chem: ChemParams { n_base: 16, ..Default::default() }, ..Default::default() };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        let mut w = World::new(&cfg);
        w.events.push(10, ExternalEvent::Temperature { cx: 5, cy: 5, radius: 2, delta: 0.5 });
        w.events.push(20, ExternalEvent::DropMatter { x: 3, y: 3, material: 5, mass: 2000 });
        w.run(100);
        state_hash(&w)
    })
}

#[test]
fn golden_state_hash_with_events_at_1_and_8_threads() {
    let a = run(1);
    let b = run(8);
    println!("GOLDEN {:#018x}", a);
    assert_eq!(a, b, "1 and 8 threads diverged");
    assert_eq!(a, GOLDEN);
}
