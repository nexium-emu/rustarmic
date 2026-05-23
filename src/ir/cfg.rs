//! Multi-block CFG glue.
//!
//! We don't run global analyses, but we do want to translate (and chain) a
//! tree of basic blocks rooted at the dispatched PC. The `Cfg` stores already-
//! translated blocks keyed by `start_pc` and tracks pending direct edges that
//! the dispatcher will patch once their targets compile.

use crate::ir::Block;
use std::collections::HashMap;

#[derive(Default)]
pub struct Cfg {
    /// Translated (and emitted) blocks keyed by guest PC.
    pub blocks: HashMap<u64, Block>,
}

impl Cfg {
    pub fn new() -> Self {
        Self { blocks: HashMap::new() }
    }

    pub fn contains(&self, pc: u64) -> bool {
        self.blocks.contains_key(&pc)
    }

    pub fn insert(&mut self, block: Block) {
        let pc = block.start_pc;
        self.blocks.insert(pc, block);
    }
}
