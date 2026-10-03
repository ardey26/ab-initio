use abi_core::chem::generate::ChemParams;
use abi_core::world::config::WorldConfig;
use abi_core::world::World;

#[test]
fn mass_is_exactly_conserved_over_1000_ticks() {
    let mut w = World::new(&WorldConfig { seed: 5, width: 64, height: 32, pop0: 400, chem: ChemParams { n_base: 16, ..Default::default() }, ..Default::default() });
    let m0 = w.conserved_mass();
    for t in 0..1000 {
        w.step();
        assert_eq!(w.conserved_mass(), m0, "mass changed at tick {}", t);
    }
    // The run must have exercised the paths whose mass accounting is being checked.
    let s = &w.stats;
    assert!(s.births > 0, "no births occurred");
    assert!(s.deaths > 0, "no deaths occurred");
    assert!(s.combines > 0, "no combines occurred");
    assert!(s.actions[0] > 0, "no moves occurred");
}

#[test]
fn zero_agent_world_runs_and_conserves() {
    let mut w = World::new(&WorldConfig { seed: 6, width: 32, height: 32, pop0: 0, ..Default::default() });
    let m0 = w.conserved_mass();
    w.run(500);
    assert_eq!(w.conserved_mass(), m0);
    assert!(w.stats.env_reactions > 0 || w.chem.table.len() == w.chem.table.n_base());
}
