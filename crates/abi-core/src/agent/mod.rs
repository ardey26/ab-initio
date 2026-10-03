pub mod actions;
pub mod brain;
pub mod genome;
pub mod memory;
pub mod sensors;

use crate::chem::props::MatId;
use genome::Genome;
use memory::SocialMemory;
use serde::{Deserialize, Serialize};

pub const BODY_TARGET: u32 = 500;
pub const BODY_REPRO_EXTRA: u32 = 500;
pub const TAKE_MAX: u32 = 1000;
pub const DIGEST_RATE: u32 = 100;
pub const ENERGY_PER_UNIT: f32 = 150.0;
pub const BASE_COST: f32 = 1.0;
pub const COGNITION_COST_PER_WEIGHT: f32 = 0.0002;
pub const REPRO_ENERGY: f32 = 250.0;
pub const START_ENERGY: f32 = 200.0;
pub const LIFESPAN: u32 = 1500;
pub const HEAT_DELTA: f32 = 0.2;
pub const STRIKE_YIELD: u32 = 500;
pub const STRIKE_DAMAGE: f32 = 15.0;
pub const ACTION_COST: [f32; 8] = [0.3, 0.2, 0.2, 1.0, 3.0, 2.0, 0.2, 0.1];
pub const NO_ACTION: u8 = 255;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Agent {
    pub alive: bool,
    pub id: u64,
    pub parent: u64,
    pub x: u16,
    pub y: u16,
    pub energy: f32,
    pub age: u32,
    pub body: u32,
    pub held: [(MatId, u32); 2],
    pub signal: [f32; 2],
    pub last_action: u8,
    pub genome: Genome,
    pub memory: SocialMemory,
}

impl Agent {
    pub fn body_target(&self) -> u32 {
        (BODY_TARGET as f32 * self.genome.body_size) as u32
    }
    pub fn free_slot(&self) -> Option<usize> {
        self.held.iter().position(|h| h.1 == 0)
    }
}
