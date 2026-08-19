pub mod code_cache;
pub mod context;
pub mod dispatcher;
pub mod memory;

pub use context::CpuContext;
pub use memory::{FlatMemory, Memory};

use crate::error::{Error, Result};
use crate::frontend::{TranslateOptions, translate_block_into};
use crate::ir::Block;
use crate::optimizer::{Scratch, optimize_with_scratch};

use code_cache::CodeCache;
use dispatcher::JitFn;

pub use crate::backend::cpu_features::CpuFeatures;

#[derive(Clone, Debug)]
pub struct JitConfig {
    pub translate: TranslateOptions,
    pub code_cache_bytes: usize,
    pub use_fastmem: bool,
    /// `None` uses runtime CPUID.  Tests and embedders may provide a masked
    /// feature set to exercise portable fallbacks deterministically.
    pub host_features: Option<CpuFeatures>,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            translate: TranslateOptions::default(),
            code_cache_bytes: 256 * 1024 * 1024,
            use_fastmem: false,
            host_features: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitReason {
    Stopped,
    BudgetExhausted,
    Svc(u32),
    Brk(u32),
    Hvc(u32),
    MemoryFault(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryAccess {
    Read,
    Write,
    Execute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryFaultCause {
    Unmapped,
    Permission,
    Alignment,
    Overflow,
    Host,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryFault {
    pub pc: u64,
    pub address: u64,
    pub size: u8,
    pub access: MemoryAccess,
    pub cause: MemoryFaultCause,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnsupportedInfo {
    pub pc: u64,
    pub opcode: u32,
    pub decoded_class: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    BudgetExhausted,
    Halted,
    Svc(u32),
    Brk(u32),
    Hvc(u32),
    Yield,
    Wait,
    MemoryFault(MemoryFault),
    Unsupported(UnsupportedInfo),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunOutcome {
    /// Compatibility view used by the original low-level embedding API.
    pub reason: ExitReason,
    /// Structured stop reason for new embedders.
    pub stop: StopReason,
    pub retired: u64,
}

impl RunOutcome {
    fn from_reason(reason: ExitReason, retired: u64) -> Self {
        let stop = match reason {
            ExitReason::BudgetExhausted => StopReason::BudgetExhausted,
            ExitReason::Svc(imm) => StopReason::Svc(imm),
            ExitReason::Brk(imm) => StopReason::Brk(imm),
            ExitReason::Hvc(imm) => StopReason::Hvc(imm),
            ExitReason::MemoryFault(address) => StopReason::MemoryFault(MemoryFault {
                pc: 0,
                address,
                size: 0,
                access: MemoryAccess::Read,
                cause: MemoryFaultCause::Unknown,
            }),
            ExitReason::Stopped => StopReason::Halted,
        };
        Self {
            reason,
            stop,
            retired,
        }
    }
}

pub struct Jit {
    pub cache: CodeCache,
    pub scratch: Scratch,
    pub block: Block,
    pub config: JitConfig,
    pub host_features: CpuFeatures,
}

impl Jit {
    pub fn new(config: JitConfig) -> Result<Self> {
        let host_features = config
            .host_features
            .unwrap_or_else(crate::backend::cpu_features::detect_features);
        if !host_features.has_sse41 {
            return Err(Error::UnsupportedHost);
        }
        Ok(Self {
            cache: CodeCache::new(config.code_cache_bytes)?,
            scratch: Scratch::new(),
            block: Block::new(0),
            config,
            host_features,
        })
    }

    pub fn invalidate_range(&mut self, start: u64, len: u64) {
        self.cache.invalidate_range(start, len);
    }

    pub fn run(&mut self, ctx: &mut CpuContext, mem: &mut dyn Memory) -> Result<ExitReason> {
        Ok(self.run_bounded(ctx, mem, u64::MAX)?.reason)
    }

    /// Execute at most `budget` guest instructions.  Blocks which do not fit
    /// in the remaining budget are compiled as transient capped blocks, so a
    /// one-instruction step never silently executes a whole basic block.
    pub fn run_bounded(
        &mut self,
        ctx: &mut CpuContext,
        mem: &mut dyn Memory,
        budget: u64,
    ) -> Result<RunOutcome> {
        let mut retired = 0u64;
        // Fault state belongs to one run invocation.  Helpers publish it
        // before returning; clear stale state before the next guest slice.
        ctx.mem_fault = 0;
        loop {
            // Avoid dereferencing a raw stop-token pointer here; callers can
            // still request stop via `ctx.should_halt`.
            let token_halt = false;
            if ctx.should_halt != 0 || token_halt {
                ctx.should_halt = 0;
                return Ok(RunOutcome::from_reason(ExitReason::Stopped, retired));
            }
            if retired >= budget {
                return Ok(RunOutcome::from_reason(
                    ExitReason::BudgetExhausted,
                    retired,
                ));
            }
            let pc = ctx.pc;
            let remaining = budget - retired;
            let cached = self.cache.lookup_meta(pc);
            let cached_fits = cached.is_some_and(|(_, count)| u64::from(count) <= remaining);
            let host_fn = if cached_fits {
                cached.unwrap().0
            } else {
                let mut translate = self.config.translate;
                let cache_block = remaining >= u64::from(translate.max_insts.max(1));
                if !cache_block {
                    translate.max_insts = remaining.min(u64::from(u32::MAX)) as u32;
                    translate.max_insts = translate.max_insts.max(1);
                }
                let translation = translate_block_into(
                    &mut self.block,
                    pc,
                    &mut |addr| mem.fetch_inst(addr),
                    translate,
                );
                if let Err(error) = translation {
                    return match error {
                        Error::GuestMemory { addr } => Ok(RunOutcome {
                            reason: ExitReason::MemoryFault(addr),
                            stop: StopReason::MemoryFault(MemoryFault {
                                pc,
                                address: addr,
                                size: 4,
                                access: MemoryAccess::Execute,
                                cause: MemoryFaultCause::Unmapped,
                            }),
                            retired,
                        }),
                        Error::Unsupported {
                            pc: error_pc,
                            opcode,
                        }
                        | Error::Decode {
                            pc: error_pc,
                            opcode,
                        } => {
                            Ok(RunOutcome {
                                reason: ExitReason::Stopped,
                                stop: StopReason::Unsupported(UnsupportedInfo {
                                    // Translation can walk past the block
                                    // entry before finding the unsupported
                                    // instruction. Preserve that exact guest
                                    // PC instead of reporting the block start.
                                    pc: error_pc,
                                    opcode,
                                    decoded_class: disarm64::decoder::decode(opcode)
                                        .map(|decoded| format!("{:?}", decoded.operation))
                                        .unwrap_or_else(|| "decode".to_string()),
                                }),
                                retired,
                            })
                        }
                        other => Err(other),
                    };
                }
                optimize_with_scratch(&mut self.block, &mut self.scratch);
                self.block.use_fastmem = self.config.use_fastmem;
                let emitted =
                    crate::backend::cpu_features::with_features(self.host_features, || {
                        crate::backend::emit_block(&self.block)
                    })?;
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
                let installed = match self.cache.install(
                    pc,
                    self.block.end_pc,
                    self.block.cycles,
                    &emitted.code,
                    &emitted.chains,
                    emitted.body_offset,
                    cache_block,
                ) {
                    Ok(ptr) => ptr,
                    Err(Error::CodeCacheFull) => {
                        self.cache.reset()?;
                        self.cache.install(
                            pc,
                            self.block.end_pc,
                            self.block.cycles,
                            &emitted.code,
                            &emitted.chains,
                            emitted.body_offset,
                            cache_block,
                        )?
                    }
                    Err(error) => return Err(error),
                };
                // Transient capped blocks are not inserted in the lookup
                // table, but their emitted pointer remains valid until the
                // next cache epoch reset.
                if cache_block {
                    self.cache.lookup(pc).ok_or(Error::CodeCacheFull)?
                } else {
                    installed
                }
            };

            let block_insns = if cached_fits {
                cached.unwrap().1
            } else {
                self.block.cycles.max(1)
            };

            let next_pc = unsafe {
                let thunk: JitFn = core::mem::transmute(self.cache.thunk());
                thunk(host_fn as u64, ctx as *mut CpuContext)
            };

            if ctx.mem_fault != 0 {
                retired = retired.saturating_add(u64::from(block_insns));
                let access = match ctx.mem_fault_access {
                    1 => MemoryAccess::Write,
                    2 => MemoryAccess::Execute,
                    _ => MemoryAccess::Read,
                };
                let cause = match ctx.mem_fault_cause {
                    1 => MemoryFaultCause::Permission,
                    2 => MemoryFaultCause::Alignment,
                    3 => MemoryFaultCause::Overflow,
                    4 => MemoryFaultCause::Host,
                    _ => MemoryFaultCause::Unmapped,
                };
                return Ok(RunOutcome {
                    reason: ExitReason::MemoryFault(ctx.mem_fault_addr),
                    stop: StopReason::MemoryFault(MemoryFault {
                        pc: ctx.mem_fault_pc,
                        address: ctx.mem_fault_addr,
                        size: ctx.mem_fault_size,
                        access,
                        cause,
                    }),
                    retired,
                });
            }

            #[cfg(feature = "tracing")]
            log::trace!(
                target: "rustarmic::jit",
                "exit pc={pc:#x} next={next_pc:#x}",
            );

            if (next_pc >> 60) == 0xE {
                let kind = next_pc & 0xFF;
                let imm = ((next_pc >> 8) & 0xFFFF) as u32;
                retired = retired.saturating_add(u64::from(block_insns));
                let reason = match kind {
                    0x01 => ExitReason::Svc(imm),
                    0x02 => ExitReason::Brk(imm),
                    0x03 => ExitReason::Hvc(imm),
                    _ => ExitReason::Stopped,
                };
                return Ok(RunOutcome::from_reason(reason, retired));
            }
            if next_pc == u64::MAX {
                retired = retired.saturating_add(u64::from(block_insns));
                return Ok(RunOutcome::from_reason(ExitReason::Stopped, retired));
            }
            retired = retired.saturating_add(u64::from(block_insns));
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
