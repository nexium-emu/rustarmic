use iced_x86::code_asm::*;

use crate::backend::abi::CTX_REG;
use crate::error::{Error, Result};

pub const BLOCK_SCRATCH_BYTES: i32 = 4096;

#[cfg(target_os = "windows")]
const XMM_SAVE_BYTES: i32 = 160;

#[cfg(not(target_os = "windows"))]
const XMM_SAVE_BYTES: i32 = 0;

const fn frame_size() -> i32 {
    let raw = BLOCK_SCRATCH_BYTES + XMM_SAVE_BYTES;
    let rem = raw % 16;
    if rem == 8 { raw } else { raw + (8 + 16 - rem) % 16 }
}

const THUNK_FRAME: i32 = frame_size();

#[cfg(target_os = "windows")]
const ARG_BLOCK: AsmRegister64 = rcx;
#[cfg(target_os = "windows")]
const ARG_CTX:   AsmRegister64 = rdx;

#[cfg(not(target_os = "windows"))]
const ARG_BLOCK: AsmRegister64 = rdi;
#[cfg(not(target_os = "windows"))]
const ARG_CTX:   AsmRegister64 = rsi;

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
