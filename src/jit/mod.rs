//! Public JIT facade: holds the code cache, owns the dispatch loop, exposes
//! `run()` to the embedder.

pub mod code_cache;
pub mod context;
pub mod memory;
pub mod dispatcher;

pub use context::CpuContext;
pub use memory::Memory;

use crate::error::{Error, Result};
use crate::frontend::{translate_block, TranslateOptions};
use crate::ir::Cfg;
use crate::optimizer::{optimize_with_scratch, Scratch};

use code_cache::CodeCache;
use dispatcher::JitFn;

#[derive(Clone, Debug)]
pub struct JitConfig {
    pub translate: TranslateOptions,
    /// Initial allocation size of the executable code cache, in bytes.
    pub code_cache_bytes: usize,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            translate: TranslateOptions::default(),
            code_cache_bytes: 16 * 1024 * 1024,
        }
    }
}

/// Reason `Jit::run` returned to the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitReason {
    /// Guest hit an unhandled instruction or block budget.
    Stopped,
    /// SVC #imm.
    Svc(u32),
    /// BRK #imm.
    Brk(u32),
    /// HVC #imm.
    Hvc(u32),
    /// Guest memory access failed.
    MemoryFault(u64),
}

pub struct Jit {
    pub cfg: Cfg,
    pub cache: CodeCache,
    pub scratch: Scratch,
    pub config: JitConfig,
}

impl Jit {
    pub fn new(config: JitConfig) -> Result<Self> {
        Ok(Self {
            cfg: Cfg::new(),
            cache: CodeCache::new(config.code_cache_bytes)?,
            scratch: Scratch::new(),
            config,
        })
    }

    /// Compile (if needed) and execute the block at `ctx.pc`. Returns the
    /// post-execution `ExitReason`.
    ///
    /// `mem` provides the embedder's view of guest memory: both for instruction
    /// fetch during translation and for the JITted code's data accesses (via
    /// `ctx.mem_base`).
    pub fn run(&mut self, ctx: &mut CpuContext, mem: &mut dyn Memory) -> Result<ExitReason> {
        loop {
            let pc = ctx.pc;
            let host_fn = if let Some(p) = self.cache.lookup(pc) {
                p
            } else {
                let block = translate_block(pc, &mut |addr| mem.fetch_inst(addr), self.config.translate)?;
                let mut block = block;
                optimize_with_scratch(&mut block, &mut self.scratch);
                let emitted = crate::backend::emit_block(&block)?;
                let host_ptr = self.cache.install(pc, &emitted.code)?;
                self.cfg.insert(block);
                host_ptr
            };

            // SAFETY: host_fn was produced by us, code memory is RWX, and the
            // calling convention matches CpuContext layout.
            let next_pc = unsafe {
                let f: JitFn = core::mem::transmute(host_fn);
                f(ctx as *mut CpuContext)
            };

            // Decode the sentinel exception encoding, if any.
            if (next_pc >> 60) == 0xE {
                let kind = next_pc & 0xFF;
                return Ok(match kind {
                    0x01 => ExitReason::Svc(0),
                    0x02 => ExitReason::Brk(0),
                    0x03 => ExitReason::Hvc(0),
                    _    => ExitReason::Stopped,
                });
            }
            if next_pc == u64::MAX {
                return Ok(ExitReason::Stopped);
            }
            ctx.pc = next_pc;
        }
    }
}

impl core::convert::From<crate::error::Error> for ExitReason {
    fn from(e: crate::error::Error) -> Self {
        match e {
            Error::GuestMemory { addr } => ExitReason::MemoryFault(addr),
            _ => ExitReason::Stopped,
        }
    }
}
