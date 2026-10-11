pub mod atomic;
pub mod code_cache;
pub mod context;
pub mod dispatcher;
pub mod helpers;
pub mod memory;
pub mod page_table;

pub use context::CpuContext;
pub use memory::{FlatMemory, Memory};

use crate::backend::thunk::ThunkKind;
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
    /// Page-table fastmem.  `Some(bits)` makes loads/stores whose address is
    /// below `1 << bits`, does not cross a 4 KiB page, and whose page has
    /// the matching `PAGE_DIRECT_*` bit in `CpuContext::page_table` access
    /// `CpuContext::mem_base + address` inline.  Everything else calls the
    /// memory callbacks, so the embedder keeps full control over unmapped
    /// pages, permission faults, self-modifying code and watchpoints by
    /// leaving those pages' bits clear.  The table must cover
    /// `1 << (bits - 12)` bytes.  Takes precedence over `use_fastmem`.
    pub fastmem_page_bits: Option<u8>,
    /// Patch direct-branch exits to jump straight into their (cached) target
    /// blocks and let indirect branches probe the dispatch table in
    /// generated code.  `false` returns to the dispatcher after every block,
    /// as before linking existed.  Budget and halt checks are identical
    /// either way.
    pub link_blocks: bool,
    /// `None` uses runtime CPUID.  Tests and embedders may provide a masked
    /// feature set to exercise portable fallbacks deterministically.
    pub host_features: Option<CpuFeatures>,
    /// After a memory fault, leave `ctx.pc` where re-entering continues
    /// past the faulting access instead of retrying it, the way dynarmic
    /// treats an access its unmapped-memory callback answered: the access
    /// is dropped (a load reads 0) and the fault is still reported; a later
    /// callback access the block skipped because of the fault
    /// (`CpuContext::mem_fault_skip_pc`) runs on re-entry, and a block that
    /// ended in SVC or HVC re-runs that call.  With `false` (the default)
    /// `ctx.pc` is the last callback access the block reached, so
    /// re-entering an unchanged mapping faults again.
    pub resume_after_fault: bool,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            translate: TranslateOptions::default(),
            code_cache_bytes: 256 * 1024 * 1024,
            use_fastmem: false,
            fastmem_page_bits: None,
            link_blocks: true,
            host_features: None,
            resume_after_fault: false,
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
    /// Test-only deliberate miscompilation applied to blocks this `Jit`
    /// translates from now on (see `frontend::fault_injection`).
    #[cfg(feature = "fault-injection")]
    pub fault_injection: Option<crate::frontend::fault_injection::FaultInjection>,
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
            #[cfg(feature = "fault-injection")]
            fault_injection: None,
        })
    }

    pub fn invalidate_range(&mut self, start: u64, len: u64) {
        self.cache.invalidate_range(start, len);
    }

    pub fn run(&mut self, ctx: &mut CpuContext, mem: &mut dyn Memory) -> Result<ExitReason> {
        Ok(self.run_bounded(ctx, mem, u64::MAX)?.reason)
    }

    /// Translates, compiles and publishes the block at `pc`.  `cap` limits
    /// the block to that many instructions and keeps it out of the dispatch
    /// table and the link graph (budget-capped blocks for small budgets).
    fn compile(
        &mut self,
        pc: u64,
        mem: &mut dyn Memory,
        cap: Option<u32>,
    ) -> Result<(*const u8, ThunkKind)> {
        let mut translate = self.config.translate;
        if let Some(cap) = cap {
            translate.max_insts = translate.max_insts.min(cap).max(1);
        }
        #[cfg(feature = "fault-injection")]
        let _injection = crate::frontend::fault_injection::scope(self.fault_injection);
        loop {
            translate_block_into(
                &mut self.block,
                pc,
                &mut |addr| mem.fetch_inst(addr),
                translate,
            )?;
            optimize_with_scratch(&mut self.block, &mut self.scratch);
            self.block.use_fastmem = self.config.use_fastmem;
            self.block.fastmem_page_bits = self.config.fastmem_page_bits.unwrap_or(0);
            self.block.link_indirect = self.config.link_blocks;
            let emitted = crate::backend::cpu_features::with_features(self.host_features, || {
                crate::backend::emit_block(&self.block)
            });
            let emitted = match emitted {
                Ok(emitted) => emitted,
                // Too many live values for the spill frame: retry shorter.
                Err(Error::BlockTooLarge { .. }) if translate.max_insts > 1 => {
                    translate.max_insts /= 2;
                    continue;
                }
                Err(error) => return Err(error),
            };
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
            // Keyed by the *requested* cap, which is what `run_bounded` looks
            // up.  The block itself may be shorter (BlockTooLarge retry, or
            // a terminator before the cap); keying by its length would make
            // every later lookup miss and leak a fresh copy per call.
            let cap_key = cap;
            let chains: &[crate::backend::ChainSite] = if self.config.link_blocks {
                &emitted.chains
            } else {
                &[]
            };
            let install = |cache: &mut CodeCache, block: &Block| {
                cache.install_block(
                    pc,
                    &block.ranges,
                    block.cycles,
                    &emitted.code,
                    chains,
                    cap_key,
                    emitted.thunk_kind,
                )
            };
            let host = match install(&mut self.cache, &self.block) {
                Err(Error::CodeCacheFull) => {
                    // Only reached from the dispatcher loop, never while
                    // generated code is on the stack.
                    self.cache.reset()?;
                    install(&mut self.cache, &self.block)
                }
                other => other,
            }?;
            return Ok((host, emitted.thunk_kind));
        }
    }

    /// Execute at most `budget` guest instructions.  Every generated block
    /// charges its instruction count against `ctx.budget_remaining` on entry
    /// and refuses to run (returning its own PC) when it does not fit, so
    /// chained blocks honour the budget too.  When a block does not fit
    /// after some progress was made the slice simply ends early; only a
    /// slice that could not otherwise make progress (e.g. single-stepping)
    /// compiles a budget-capped block, and those are cached per (pc, cap).
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
        ctx.mem_fault_skip_pc = 0;
        loop {
            if ctx.should_halt != 0 {
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
            let slice = remaining.min(i64::MAX as u64);

            let (mut host_fn, mut thunk_kind) = match self.cache.lookup_dispatch_meta(pc) {
                Some(entry) => (entry.host_ptr, entry.thunk_kind),
                None => match self.compile(pc, mem, None) {
                    Ok(compiled) => compiled,
                    Err(error) => return translation_stop(error, pc, retired),
                },
            };
            let mut capped = false;
            let (next_pc, executed) = loop {
                // Both are stable across cache resets, but cheap to refresh.
                ctx.dispatch_table = self.cache.dispatch_table().cast();
                let entry_thunk = if self.config.link_blocks {
                    ThunkKind::Full
                } else {
                    thunk_kind
                };
                let thunk: JitFn =
                    unsafe { core::mem::transmute(self.cache.thunk_for(entry_thunk)) };
                ctx.budget_remaining = slice as i64;
                let next_pc = unsafe { thunk(host_fn as u64, ctx as *mut CpuContext) };
                let executed = slice - ctx.budget_remaining.max(0) as u64;
                let refused = executed == 0
                    && next_pc == pc
                    && ctx.should_halt == 0
                    && ctx.mem_fault == 0;
                if !refused || capped {
                    break (next_pc, executed);
                }
                // The block at `pc` is longer than the remaining budget.
                if retired > 0 {
                    return Ok(RunOutcome::from_reason(
                        ExitReason::BudgetExhausted,
                        retired,
                    ));
                }
                let cap = remaining.min(u64::from(u32::MAX)) as u32;
                (host_fn, thunk_kind) = match self.cache.lookup_capped(pc, cap) {
                    Some(host) => (host, ThunkKind::Full),
                    None => match self.compile(pc, mem, Some(cap)) {
                        Ok(compiled) => compiled,
                        Err(error) => return translation_stop(error, pc, retired),
                    },
                };
                capped = true;
            };
            retired = retired.saturating_add(executed);

            if ctx.mem_fault != 0 {
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
                if self.config.resume_after_fault {
                    ctx.pc = fault_resume_pc(ctx.pc, ctx.mem_fault_skip_pc, next_pc);
                }
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
                let reason = match kind {
                    0x01 => ExitReason::Svc(imm),
                    0x02 => ExitReason::Brk(imm),
                    0x03 => ExitReason::Hvc(imm),
                    _ => ExitReason::Stopped,
                };
                return Ok(RunOutcome::from_reason(reason, retired));
            }
            if next_pc == u64::MAX {
                return Ok(RunOutcome::from_reason(ExitReason::Stopped, retired));
            }
            ctx.pc = next_pc;
        }
    }
}

/// Where `JitConfig::resume_after_fault` re-enters after a memory fault.
/// Once an access faults the block keeps running to its end with later
/// callback accesses skipped; `skip_pc` is the first of those (0 when none
/// was), which has not happened, so execution resumes there.  Otherwise
/// the whole block ran: continue with what it returned (`next_pc`).  An
/// SVC/HVC/BRK terminal's exit is discarded by the fault report, so its
/// instruction is re-run: SVC/HVC store `pc + 4` in `ctx_pc`, BRK its own
/// address.
fn fault_resume_pc(ctx_pc: u64, skip_pc: u64, next_pc: u64) -> u64 {
    if skip_pc != 0 {
        return skip_pc;
    }
    if (next_pc >> 60) == 0xE {
        return match next_pc & 0xFF {
            0x01 | 0x03 => ctx_pc.wrapping_sub(4),
            _ => ctx_pc,
        };
    }
    if next_pc == u64::MAX {
        return ctx_pc;
    }
    next_pc
}

/// Maps a translation failure at `pc` to the structured stop the embedder
/// sees; other errors propagate.
fn translation_stop(error: Error, pc: u64, retired: u64) -> Result<RunOutcome> {
    match error {
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
        } => Ok(RunOutcome {
            reason: ExitReason::Stopped,
            stop: StopReason::Unsupported(UnsupportedInfo {
                // Translation can walk past the block entry before finding
                // the unsupported instruction. Preserve that exact guest PC
                // instead of reporting the block start.
                pc: error_pc,
                opcode,
                decoded_class: disarm64::decoder::decode(opcode)
                    .map(|decoded| format!("{:?}", decoded.operation))
                    .unwrap_or_else(|| "decode".to_string()),
            }),
            retired,
        }),
        other => Err(other),
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

#[cfg(test)]
mod tests {
    use super::*;

    struct Nops;
    impl Memory for Nops {
        fn fetch_inst(&mut self, _addr: u64) -> Option<u32> {
            Some(0xD503_201F)
        }
    }

    #[test]
    fn fault_resume_pc_moves_past_the_faulting_access() {
        // Nothing skipped: the block ran to its end, continue after it.
        assert_eq!(fault_resume_pc(0x1008, 0, 0x2000), 0x2000);
        // A later access was skipped: it has not run yet, resume there.
        assert_eq!(fault_resume_pc(0x1010, 0x100C, 0x2000), 0x100C);
        // Dispatcher stop sentinel: nothing better than the stored PC.
        assert_eq!(fault_resume_pc(0x1008, 0, u64::MAX), 0x1008);
        // SVC / HVC terminal stored pc + 4: re-run the call itself.
        let svc = 0xE000_0000_0000_0000 | (5 << 8) | 0x01;
        let hvc = 0xE000_0000_0000_0000 | 0x03;
        assert_eq!(fault_resume_pc(0x1010, 0, svc), 0x100C);
        assert_eq!(fault_resume_pc(0x1010, 0, hvc), 0x100C);
        // ...unless an access before it was skipped.
        assert_eq!(fault_resume_pc(0x1010, 0x1004, svc), 0x1004);
        // BRK stores its own address.
        let brk = 0xE000_0000_0000_0000 | 0x02;
        assert_eq!(fault_resume_pc(0x100C, 0, brk), 0x100C);
    }

    /// A capped block is cached under the cap `run_bounded` looks it up
    /// with, even when the translated block is shorter than the cap (the
    /// `BlockTooLarge` retry, or a configured `max_insts` below the cap).
    /// Keying it by its own length made every later lookup miss and leak a
    /// fresh copy per call.
    #[test]
    fn capped_block_is_found_under_requested_cap() {
        let mut config = JitConfig::default();
        config.translate.max_insts = 4;
        config.code_cache_bytes = 1 << 20;
        let mut jit = Jit::new(config).unwrap();
        let mut mem = Nops;
        let (host, _) = jit.compile(0x1000, &mut mem, Some(9)).unwrap();
        assert_eq!(jit.block.cycles, 4);
        assert_eq!(jit.cache.lookup_capped(0x1000, 9), Some(host));
        let used = jit.cache.used_bytes();
        for _ in 0..8 {
            if jit.cache.lookup_capped(0x1000, 9).is_none() {
                jit.compile(0x1000, &mut mem, Some(9)).unwrap();
            }
        }
        assert_eq!(jit.cache.used_bytes(), used, "capped block retranslated");
    }
}
