//! One-time host→JIT thunk.
//!
//! Every block used to emit its own push/pop sequence for the callee-saved
//! GPRs (RBX, R12-R15) and a save loop for the callee-saved XMMs it touched.
//! That's 20+ bytes of dead overhead at the head and tail of every chain.
//!
//! With a shared thunk we pay that prologue cost once per `Jit::run`
//! iteration and every block's body is just the body — no push/pop, no
//! XMM save, no per-block rsp bump. Blocks always run with R15 already
//! holding the CpuContext pointer and RSP pointing into a pre-allocated
//! scratch area so spill slots `[rbp - off]` Just Work.
//!
//! Calling convention: the thunk itself is `extern "C" fn(block_fn: u64,
//! ctx: *mut CpuContext) -> u64` — arg0 is the first block to run, arg1
//! is the CPU context. Blocks return their next PC (or an exit token) in
//! RAX, which falls back through the thunk to the host caller.

use iced_x86::code_asm::*;

use crate::backend::abi::CTX_REG;
use crate::error::{Error, Result};

/// Bytes reserved between rbp and the start of per-block spill space.
/// Layout (from rbp downward):
///   [rbp - 8 .. rbp - 40]  = saved RBX, R12-R15 (5 pushes)
///   [rbp - 48 .. rbp - X]  = block spill scratch (4096 bytes on Win/SysV)
///   [rsp + 0 .. 160]       = XMM6..XMM15 saves (Windows only)
///   [rsp + 160 .. 4264]    = (covered by spill scratch above on Win)
///
/// Block regalloc places its first spill at `[rbp - SPILL_FIRST]`, where
/// `SPILL_FIRST = SAVED_SIZE + 8`. We keep `SAVED_SIZE = 40` so the regalloc
/// arithmetic and existing tests stay untouched.
pub const BLOCK_SCRATCH_BYTES: i32 = 4096;

#[cfg(target_os = "windows")]
const XMM_SAVE_BYTES: i32 = 160; // 10 XMM regs (xmm6..xmm15), 16 bytes each

#[cfg(not(target_os = "windows"))]
const XMM_SAVE_BYTES: i32 = 0;   // SysV has no callee-saved XMM regs

/// Total frame size the thunk subtracts from rsp after its callee-saved
/// pushes. We need this to be 8 more than a multiple of 16, so that the
/// initial 8-byte misalignment from the 5 callee-saved pushes is corrected
/// to leave rsp 16-aligned at the call site.
const fn frame_size() -> i32 {
    // 5 callee-saved GPR pushes from a 16-aligned base leave us misaligned
    // by 8. To realign for the host ABI call site we need (frame_size mod
    // 16) == 8.
    let raw = BLOCK_SCRATCH_BYTES + XMM_SAVE_BYTES;
    let rem = raw % 16;
    if rem == 8 { raw } else { raw + (8 + 16 - rem) % 16 }
}

const THUNK_FRAME: i32 = frame_size();

#[cfg(target_os = "windows")]
const ARG_BLOCK: AsmRegister64 = rcx; // arg0
#[cfg(target_os = "windows")]
const ARG_CTX:   AsmRegister64 = rdx; // arg1

#[cfg(not(target_os = "windows"))]
const ARG_BLOCK: AsmRegister64 = rdi;
#[cfg(not(target_os = "windows"))]
const ARG_CTX:   AsmRegister64 = rsi;

/// Assemble the thunk machine code. The returned bytes are position-
/// independent — the caller copies them into the rwx code cache.
pub fn emit_thunk_bytes() -> Result<Vec<u8>> {
    let mut asm = CodeAssembler::new(64).map_err(into_err)?;

    asm.push(rbp).map_err(into_err)?;
    asm.mov(rbp, rsp).map_err(into_err)?;
    asm.push(rbx).map_err(into_err)?;
    asm.push(r12).map_err(into_err)?;
    asm.push(r13).map_err(into_err)?;
    asm.push(r14).map_err(into_err)?;
    asm.push(r15).map_err(into_err)?;

    asm.sub(rsp, THUNK_FRAME).map_err(into_err)?;

    #[cfg(target_os = "windows")]
    {
        // Save callee-saved XMMs at the low end of the scratch (closest to rsp).
        for i in 0..10i32 {
            let xmm_reg = match i {
                0 => xmm6, 1 => xmm7, 2 => xmm8, 3 => xmm9, 4 => xmm10,
                5 => xmm11, 6 => xmm12, 7 => xmm13, 8 => xmm14, 9 => xmm15,
                _ => unreachable!(),
            };
            asm.movdqu(xmmword_ptr(rsp + i * 16), xmm_reg).map_err(into_err)?;
        }
    }

    asm.mov(CTX_REG, ARG_CTX).map_err(into_err)?;
    asm.call(ARG_BLOCK).map_err(into_err)?;

    #[cfg(target_os = "windows")]
    {
        for i in 0..10i32 {
            let xmm_reg = match i {
                0 => xmm6, 1 => xmm7, 2 => xmm8, 3 => xmm9, 4 => xmm10,
                5 => xmm11, 6 => xmm12, 7 => xmm13, 8 => xmm14, 9 => xmm15,
                _ => unreachable!(),
            };
            asm.movdqu(xmm_reg, xmmword_ptr(rsp + i * 16)).map_err(into_err)?;
        }
    }

    asm.add(rsp, THUNK_FRAME).map_err(into_err)?;
    asm.pop(r15).map_err(into_err)?;
    asm.pop(r14).map_err(into_err)?;
    asm.pop(r13).map_err(into_err)?;
    asm.pop(r12).map_err(into_err)?;
    asm.pop(rbx).map_err(into_err)?;
    asm.pop(rbp).map_err(into_err)?;
    asm.ret().map_err(into_err)?;

    let asm_bytes = asm.assemble(0).map_err(into_err)?;
    Ok(asm_bytes)
}

fn into_err(e: iced_x86::IcedError) -> Error {
    Error::Backend(e.to_string())
}
