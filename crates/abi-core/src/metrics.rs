use crate::stats::Stats;
use crate::world::World;

pub struct MetricsRow {
    pub tick: u64,
    pub pop: usize,
    pub births: u64,
    pub deaths: u64,
    pub ticks_per_s: f64,
    pub us_per_agent_step: f64,
    pub action_rates: [f64; 8],
    pub combines: u64,
    pub combines_hot: u64,
    pub strikes_mine: u64,
    pub strikes_hit: u64,
    pub gives: u64,
    pub artifact_energy_frac: f64,
    pub distinct_materials: usize,
    pub distinct_behaviours: usize,
    pub n_materials_total: usize,
    pub env_reactions: u64,
    pub env_hot: u64,
    pub fires: u64,
    pub mean_energy: f32,
    pub mean_hidden: f32,
}

impl MetricsRow {
    pub fn header() -> &'static str {
        "tick,pop,births,deaths,ticks_per_s,us_per_agent_step,move,take,drop,combine,heat,strike,give,emit,combines,combines_hot,strikes_mine,strikes_hit,gives,artifact_energy_frac,distinct_materials,distinct_behaviours,n_materials_total,env_reactions,env_hot,fires,mean_energy,mean_hidden"
    }

    pub fn from(w: &World, s: &Stats, window: u64, seconds: f64) -> Self {
        let steps = s.agent_steps.max(1) as f64;
        let mut action_rates = [0f64; 8];
        for i in 0..8 {
            action_rates[i] = s.actions[i] as f64 / steps;
        }
        let alive: Vec<&crate::agent::Agent> = w.agents.iter().filter(|a| a.alive).collect();
        let pop = alive.len();
        let mean_energy = alive.iter().map(|a| a.energy).sum::<f32>() / pop.max(1) as f32;
        let mean_hidden = alive.iter().map(|a| a.genome.hidden as f32).sum::<f32>() / pop.max(1) as f32;
        MetricsRow {
            tick: w.tick,
            pop,
            births: s.births,
            deaths: s.deaths,
            ticks_per_s: window as f64 / seconds.max(1e-9),
            us_per_agent_step: seconds * 1e6 / steps,
            action_rates,
            combines: s.combines,
            combines_hot: s.combines_hot,
            strikes_mine: s.strikes_mine,
            strikes_hit: s.strikes_hit,
            gives: s.gives,
            artifact_energy_frac: if s.energy_total > 0.0 { s.energy_from_artifacts / s.energy_total } else { 0.0 },
            distinct_materials: s.distinct_in_use(),
            distinct_behaviours: s.distinct_behaviours(),
            n_materials_total: w.chem.table.len(),
            env_reactions: s.env_reactions,
            env_hot: s.env_hot,
            fires: s.fires,
            mean_energy,
            mean_hidden,
        }
    }

    pub fn to_csv(&self) -> String {
        let r: Vec<String> = self.action_rates.iter().map(|v| format!("{:.4}", v)).collect();
        format!(
            "{},{},{},{},{:.1},{:.3},{},{},{},{},{},{},{:.4},{},{},{},{},{},{},{:.1},{:.2}",
            self.tick, self.pop, self.births, self.deaths, self.ticks_per_s, self.us_per_agent_step, r.join(","),
            self.combines, self.combines_hot, self.strikes_mine, self.strikes_hit, self.gives, self.artifact_energy_frac,
            self.distinct_materials, self.distinct_behaviours, self.n_materials_total, self.env_reactions, self.env_hot, self.fires,
            self.mean_energy, self.mean_hidden
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::config::WorldConfig;
    use crate::world::World;

    #[test]
    fn row_has_same_field_count_as_header() {
        let mut w = World::new(&WorldConfig { seed: 1, width: 32, height: 32, pop0: 50, ..Default::default() });
        w.run(20);
        let stats = std::mem::take(&mut w.stats);
        let row = MetricsRow::from(&w, &stats, 20, 0.5);
        assert_eq!(row.to_csv().split(',').count(), MetricsRow::header().split(',').count());
        assert!(row.distinct_behaviours > 0);
        assert!(row.us_per_agent_step > 0.0);
    }
}
