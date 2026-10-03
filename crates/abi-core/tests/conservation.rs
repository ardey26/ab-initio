use abi_core::chem::generate::ChemParams;
use abi_core::world::config::WorldConfig;
use abi_core::world::World;

#[test]
fn mass_is_exactly_conserved_over_1000_ticks() {
    let mut w = World::new(&WorldConfig { seed: 5, width: 64, height: 32, pop0: 400, chem: ChemParams { n_base: 16, ..Default::default() }, ..Default::default() });
    let m0 = w.total_mass();
    for t in 0..1000 {
        w.step();
        assert_eq!(w.total_mass(), m0, "mass changed at tick {}", t);
    }
}

#[test]
fn zero_agent_world_runs_and_conserves() {
    let mut w = World::new(&WorldConfig { seed: 6, width: 32, height: 32, pop0: 0, ..Default::default() });
    let m0 = w.total_mass();
    w.run(500);
    assert_eq!(w.total_mass(), m0);
    assert!(w.stats.env_reactions > 0 || w.chem.table.len() == w.chem.table.n_base());
}
