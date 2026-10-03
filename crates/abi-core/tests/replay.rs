use abi_core::checkpoint;
use abi_core::chem::generate::ChemParams;
use abi_core::events::ExternalEvent;
use abi_core::hash::state_hash;
use abi_core::world::config::WorldConfig;
use abi_core::world::World;

fn cfg() -> WorldConfig {
    WorldConfig { seed: 31, width: 64, height: 32, pop0: 300, chem: ChemParams { n_base: 16, ..Default::default() }, ..Default::default() }
}

fn events() -> Vec<(u64, ExternalEvent)> {
    vec![
        (10, ExternalEvent::Rain { cx: 10, cy: 10, radius: 3, water_per_cell: 400 }),
        (20, ExternalEvent::Temperature { cx: 40, cy: 5, radius: 4, delta: 0.6 }),
        (30, ExternalEvent::DropMatter { x: 3, y: 3, material: 5, mass: 3000 }),
        (40, ExternalEvent::Impact { cx: 50, cy: 20, radius: 5 }),
        (50, ExternalEvent::SeedOrganism { x: 7, y: 7 }),
    ]
}

#[test]
fn events_are_deterministic_and_mass_ledger_holds() {
    let run = || {
        let mut w = World::new(&cfg());
        for (t, e) in events() {
            w.events.push(t, e);
        }
        let c0 = w.conserved_mass();
        w.run(100);
        assert_eq!(w.conserved_mass(), c0);
        assert!(w.external_mass > 0, "rain and drop add external mass");
        (state_hash(&w), w.population())
    };
    assert_eq!(run(), run());
}

#[test]
fn a_run_with_events_differs_from_one_without() {
    let mut a = World::new(&cfg());
    let mut b = World::new(&cfg());
    for (t, e) in events() {
        b.events.push(t, e);
    }
    a.run(60);
    b.run(60);
    assert_ne!(state_hash(&a), state_hash(&b));
}

#[test]
fn pending_events_change_the_state_hash() {
    let a = World::new(&cfg());
    let mut b = World::new(&cfg());
    assert_eq!(state_hash(&a), state_hash(&b));
    b.events.push(10, ExternalEvent::Temperature { cx: 1, cy: 1, radius: 1, delta: 0.1 });
    assert_ne!(state_hash(&a), state_hash(&b));
}

#[test]
fn checkpoint_round_trip_preserves_hash_and_continuation() {
    let dir = std::env::temp_dir().join(format!("abi-ck-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut w = World::new(&cfg());
    for (t, e) in events() {
        w.events.push(t, e);
    }
    w.run(25);
    let path = dir.join(checkpoint::filename(w.tick));
    let bytes = checkpoint::save(&w, &path).unwrap();
    assert!(bytes > 0);
    println!("checkpoint bytes (64x32, tick {}): {}", w.tick, bytes);
    let mut restored = checkpoint::load(&path).unwrap();
    assert_eq!(state_hash(&restored), state_hash(&w));
    // Continue both past events at ticks 30..50 and compare.
    w.run(50);
    restored.run(50);
    assert_eq!(state_hash(&restored), state_hash(&w));
    assert_eq!(restored.conserved_mass(), w.conserved_mass());
    std::fs::remove_dir_all(&dir).unwrap();
}
