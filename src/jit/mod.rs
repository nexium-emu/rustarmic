pub mod code_cache;
pub mod context;
pub mod memory;
pub mod dispatcher;

pub use context::CpuContext;
pub use memory::Memory;

use crate::error::{Error, Result};
use crate::frontend::{translate_block_into, TranslateOptions};
use crate::ir::Block;
use crate::optimizer::{optimize_with_scratch, Scratch};

use code_cache::CodeCache;
use dispatcher::JitFn;

#[derive(Clone, Debug)]
pub struct JitConfig {
    pub translate: TranslateOptions,
    pub code_cache_bytes: usize,
    /// Emit the bounds-checked direct-memory fast path for guest loads/stores.
    /// When `true`, the emitter wraps every `Load*`/`Store*` in a
    /// `(va - mem_base_va) + width <= mem_size`-check; in-range accesses hit
    /// `[mem_base + offset]` directly, out-of-range fall back to the fn-ptr
    /// handlers. Default `false` so the disabled case pays nothing.
    pub use_fastmem: bool,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            translate: TranslateOptions::default(),
            code_cache_bytes: 16 * 1024 * 1024,
            use_fastmem: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitReason {
    Stopped,
    Svc(u32),
    Brk(u32),
    Hvc(u32),
    MemoryFault(u64),
}

pub struct Jit {
    pub cache:    CodeCache,
    pub scratch:  Scratch,
    pub block:    Block,
    pub config:   JitConfig,
}

impl Jit {
    pub fn new(config: JitConfig) -> Result<Self> {
        Ok(Self {
            cache:   CodeCache::new(config.code_cache_bytes)?,
            scratch: Scratch::new(),
            block:   Block::new(0),
            config,
        })
    }

    /// Drop any cached blocks whose guest PC falls in `[start, start+len)`.
    /// See [`CodeCache::invalidate_range`] for the SMC caveats.
    pub fn invalidate_range(&mut self, start: u64, len: u64) {
        self.cache.invalidate_range(start, len);
    }

    pub fn run(&mut self, ctx: &mut CpuContext, mem: &mut dyn Memory) -> Result<ExitReason> {
        // Clear the cooperative halt flag on entry — callers expect each
        // run() invocation to start fresh.
        ctx.should_halt = 0;
        loop {
            // Cooperative halt point. A mem hook (or another thread sharing
            // *ctx via raw pointer) can set should_halt=1 to bail at the next
            // block boundary; this is what lets the embedder stop a runaway
            // unmapped-write loop without modifying the JIT thunk.
            if ctx.should_halt != 0 {
                ctx.should_halt = 0;
                return Ok(ExitReason::Stopped);
            }
            let pc = ctx.pc;
            let host_fn = if let Some(p) = self.cache.lookup(pc) {
                p
            } else {
                translate_block_into(
                    &mut self.block,
                    pc,
                    &mut |addr| mem.fetch_inst(addr),
                    self.config.translate,
                )?;
                optimize_with_scratch(&mut self.block, &mut self.scratch);
                self.block.use_fastmem = self.config.use_fastmem;
                let emitted = crate::backend::emit_block(&self.block)?;
                #[cfg(feature = "tracing")]
                {
                    let insns = self.block.cycles;
                    let ir_live = self.block.iter_live().count();
                    let host_bytes = emitted.code.len();
                    log::trace!(
                        target: "rustarmic::jit",
                        "compile pc={pc:#x} insns={insns} ir_live={ir_live} host_bytes={host_bytes}",
                    );
                }
                self.cache.install(pc, &emitted.code, &emitted.chains, emitted.body_offset)?
            };

            let next_pc = unsafe {
                let thunk: JitFn = core::mem::transmute(self.cache.thunk());
                thunk(host_fn as u64, ctx as *mut CpuContext)
            };

            #[cfg(feature = "tracing")]
            log::trace!(
                target: "rustarmic::jit",
                "exit pc={pc:#x} next={next_pc:#x}",
            );

            if (next_pc >> 60) == 0xE {
                let kind = next_pc & 0xFF;
                let imm  = ((next_pc >> 8) & 0xFFFF) as u32;
                return Ok(match kind {
                    0x01 => ExitReason::Svc(imm),
                    0x02 => ExitReason::Brk(imm),
                    0x03 => ExitReason::Hvc(imm),
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
