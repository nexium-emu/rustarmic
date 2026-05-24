use iced_x86::code_asm::*;

use crate::backend::regalloc::{Allocation, Loc};
use crate::error::{Error, Result};
use crate::ir::ValueRef;

pub fn gpr64(n: u8) -> AsmRegister64 {
    match n {
        0  => rax, 1  => rcx, 2  => rdx, 3  => rbx,
        4  => rsp, 5  => rbp, 6  => rsi, 7  => rdi,
        8  => r8,  9  => r9,  10 => r10, 11 => r11,
        12 => r12, 13 => r13, 14 => r14, 15 => r15,
        _ => panic!("invalid GPR encoding: {}", n),
    }
}

pub fn gpr32(n: u8) -> AsmRegister32 {
    match n {
        0  => eax,  1  => ecx,  2  => edx,  3  => ebx,
        4  => esp,  5  => ebp,  6  => esi,  7  => edi,
        8  => r8d,  9  => r9d,  10 => r10d, 11 => r11d,
        12 => r12d, 13 => r13d, 14 => r14d, 15 => r15d,
        _ => panic!("invalid GPR encoding: {}", n),
    }
}

pub fn load64(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    vr: ValueRef,
    dst: AsmRegister64,
) -> Result<()> {
    match alloc.loc(vr) {
        Loc::Reg(r) => {
            let src = gpr64(r);
            if src != dst {
                asm.mov(dst, src).map_err(into_err)?;
            }
        }
        Loc::Spill(off) => {
            asm.mov(dst, qword_ptr(rbp - off)).map_err(into_err)?;
        }
        Loc::None => return Err(Error::Backend(format!("load64 from Loc::None ({:?})", vr))),
    }
    Ok(())
}

pub fn load32(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    vr: ValueRef,
    dst: AsmRegister32,
) -> Result<()> {
    match alloc.loc(vr) {
        Loc::Reg(r) => {
            let src = gpr32(r);
            if src != dst {
                asm.mov(dst, src).map_err(into_err)?;
            }
        }
        Loc::Spill(off) => {
            asm.mov(dst, dword_ptr(rbp - off)).map_err(into_err)?;
        }
        Loc::None => return Err(Error::Backend(format!("load32 from Loc::None ({:?})", vr))),
    }
    Ok(())
}

pub fn store64(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    vr: ValueRef,
    src: AsmRegister64,
) -> Result<()> {
    match alloc.loc(vr) {
        Loc::Reg(r) => {
            let dst = gpr64(r);
            if dst != src {
                asm.mov(dst, src).map_err(into_err)?;
            }
        }
        Loc::Spill(off) => {
            asm.mov(qword_ptr(rbp - off), src).map_err(into_err)?;
        }
        Loc::None => {}
    }
    Ok(())
}

pub fn store32(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    vr: ValueRef,
    src: AsmRegister32,
) -> Result<()> {
    match alloc.loc(vr) {
        Loc::Reg(r) => {
            let dst = gpr32(r);
            if dst != src {
                asm.mov(dst, src).map_err(into_err)?;
            }
        }
        Loc::Spill(off) => {
            asm.mov(dword_ptr(rbp - off), src).map_err(into_err)?;
        }
        Loc::None => {}
    }
    Ok(())
}

pub fn load_xmm_s(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    vr: ValueRef,
    dst: AsmRegisterXmm,
) -> Result<()> {
    match alloc.loc(vr) {
        Loc::Reg(r) => asm.movd(dst, gpr32(r)).map_err(into_err)?,
        Loc::Spill(off) => asm.movd(dst, dword_ptr(rbp - off)).map_err(into_err)?,
        Loc::None => return Err(Error::Backend(format!("load_xmm_s from Loc::None ({:?})", vr))),
    }
    Ok(())
}

pub fn load_xmm_d(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    vr: ValueRef,
    dst: AsmRegisterXmm,
) -> Result<()> {
    match alloc.loc(vr) {
        Loc::Reg(r) => asm.movq(dst, gpr64(r)).map_err(into_err)?,
        Loc::Spill(off) => asm.movq(dst, qword_ptr(rbp - off)).map_err(into_err)?,
        Loc::None => return Err(Error::Backend(format!("load_xmm_d from Loc::None ({:?})", vr))),
    }
    Ok(())
}

pub fn store_xmm_s(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    vr: ValueRef,
    src: AsmRegisterXmm,
) -> Result<()> {
    match alloc.loc(vr) {
        Loc::Reg(r) => asm.movd(gpr32(r), src).map_err(into_err)?,
        Loc::Spill(off) => asm.movd(dword_ptr(rbp - off), src).map_err(into_err)?,
        Loc::None => {}
    }
    Ok(())
}

pub fn store_xmm_d(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    vr: ValueRef,
    src: AsmRegisterXmm,
) -> Result<()> {
    match alloc.loc(vr) {
        Loc::Reg(r) => asm.movq(gpr64(r), src).map_err(into_err)?,
        Loc::Spill(off) => asm.movq(qword_ptr(rbp - off), src).map_err(into_err)?,
        Loc::None => {}
    }
    Ok(())
}

fn into_err(e: iced_x86::IcedError) -> Error {
    Error::Backend(e.to_string())
}
