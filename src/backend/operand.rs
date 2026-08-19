use iced_x86::code_asm::*;

use crate::backend::regalloc::{Allocation, Loc};
use crate::error::{Error, Result};
use crate::ir::ValueRef;

pub fn gpr64(n: u8) -> AsmRegister64 {
    match n {
        0 => rax,
        1 => rcx,
        2 => rdx,
        3 => rbx,
        4 => rsp,
        5 => rbp,
        6 => rsi,
        7 => rdi,
        8 => r8,
        9 => r9,
        10 => r10,
        11 => r11,
        12 => r12,
        13 => r13,
        14 => r14,
        15 => r15,
        _ => panic!("invalid GPR encoding: {}", n),
    }
}

pub fn gpr32(n: u8) -> AsmRegister32 {
    match n {
        0 => eax,
        1 => ecx,
        2 => edx,
        3 => ebx,
        4 => esp,
        5 => ebp,
        6 => esi,
        7 => edi,
        8 => r8d,
        9 => r9d,
        10 => r10d,
        11 => r11d,
        12 => r12d,
        13 => r13d,
        14 => r14d,
        15 => r15d,
        _ => panic!("invalid GPR encoding: {}", n),
    }
}

pub fn gpr16(n: u8) -> AsmRegister16 {
    match n {
        0 => ax,
        1 => cx,
        2 => dx,
        3 => bx,
        4 => sp,
        5 => bp,
        6 => si,
        7 => di,
        8 => r8w,
        9 => r9w,
        10 => r10w,
        11 => r11w,
        12 => r12w,
        13 => r13w,
        14 => r14w,
        15 => r15w,
        _ => panic!("invalid GPR encoding: {}", n),
    }
}

pub fn gpr8(n: u8) -> AsmRegister8 {
    match n {
        0 => al,
        1 => cl,
        2 => dl,
        3 => bl,
        4 => spl,
        5 => bpl,
        6 => sil,
        7 => dil,
        8 => r8b,
        9 => r9b,
        10 => r10b,
        11 => r11b,
        12 => r12b,
        13 => r13b,
        14 => r14b,
        15 => r15b,
        _ => panic!("invalid GPR encoding: {}", n),
    }
}

pub fn xmm(n: u8) -> AsmRegisterXmm {
    match n {
        0 => xmm0,
        1 => xmm1,
        2 => xmm2,
        3 => xmm3,
        4 => xmm4,
        5 => xmm5,
        6 => xmm6,
        7 => xmm7,
        8 => xmm8,
        9 => xmm9,
        10 => xmm10,
        11 => xmm11,
        12 => xmm12,
        13 => xmm13,
        14 => xmm14,
        15 => xmm15,
        _ => panic!("invalid XMM encoding: {}", n),
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
        Loc::Xmm(x) => {
            asm.movq(dst, xmm(x)).map_err(into_err)?;
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
        Loc::Xmm(x) => {
            asm.movd(dst, xmm(x)).map_err(into_err)?;
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
        Loc::Xmm(x) => {
            asm.movq(xmm(x), src).map_err(into_err)?;
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
        Loc::Xmm(x) => {
            asm.movd(xmm(x), src).map_err(into_err)?;
        }
        Loc::Spill(off) => {
            asm.mov(dword_ptr(rbp - off), src).map_err(into_err)?;
            asm.mov(dword_ptr(rbp - (off - 4)), 0i32)
                .map_err(into_err)?;
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
        Loc::Xmm(x) => {
            let src = xmm(x);
            if src != dst {
                asm.movdqa(dst, src).map_err(into_err)?;
            }
        }
        Loc::Spill(off) => asm.movd(dst, dword_ptr(rbp - off)).map_err(into_err)?,
        Loc::None => {
            return Err(Error::Backend(format!(
                "load_xmm_s from Loc::None ({:?})",
                vr
            )));
        }
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
        Loc::Xmm(x) => {
            let src = xmm(x);
            if src != dst {
                asm.movdqa(dst, src).map_err(into_err)?;
            }
        }
        Loc::Spill(off) => asm.movq(dst, qword_ptr(rbp - off)).map_err(into_err)?,
        Loc::None => {
            return Err(Error::Backend(format!(
                "load_xmm_d from Loc::None ({:?})",
                vr
            )));
        }
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
        Loc::Xmm(x) => {
            let dst = xmm(x);
            if dst != src {
                asm.movdqa(dst, src).map_err(into_err)?;
            }
        }
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
        Loc::Xmm(x) => {
            let dst = xmm(x);
            if dst != src {
                asm.movdqa(dst, src).map_err(into_err)?;
            }
        }
        Loc::Spill(off) => asm.movq(qword_ptr(rbp - off), src).map_err(into_err)?,
        Loc::None => {}
    }
    Ok(())
}

pub fn load_xmm_q(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    vr: ValueRef,
    dst: AsmRegisterXmm,
) -> Result<()> {
    match alloc.loc(vr) {
        Loc::Xmm(x) => {
            let src = xmm(x);
            if src != dst {
                asm.movdqa(dst, src).map_err(into_err)?;
            }
        }
        Loc::Spill(off) => asm.movdqu(dst, xmmword_ptr(rbp - off)).map_err(into_err)?,
        Loc::Reg(_) | Loc::None => {
            return Err(Error::Backend(format!(
                "load_xmm_q from invalid loc ({:?})",
                vr
            )));
        }
    }
    Ok(())
}

pub fn store_xmm_q(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    vr: ValueRef,
    src: AsmRegisterXmm,
) -> Result<()> {
    match alloc.loc(vr) {
        Loc::Xmm(x) => {
            let dst = xmm(x);
            if dst != src {
                asm.movdqa(dst, src).map_err(into_err)?;
            }
        }
        Loc::Spill(off) => asm.movdqu(xmmword_ptr(rbp - off), src).map_err(into_err)?,
        Loc::Reg(_) => return Err(Error::Backend(format!("store_xmm_q to GPR loc ({:?})", vr))),
        Loc::None => {}
    }
    Ok(())
}

pub fn get_xmm_q(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    vr: ValueRef,
    fallback: AsmRegisterXmm,
) -> Result<AsmRegisterXmm> {
    match alloc.loc(vr) {
        Loc::Xmm(x) => Ok(xmm(x)),
        Loc::Spill(off) => {
            asm.movdqu(fallback, xmmword_ptr(rbp - off))
                .map_err(into_err)?;
            Ok(fallback)
        }
        Loc::Reg(_) | Loc::None => Err(Error::Backend(format!(
            "get_xmm_q from invalid loc ({:?})",
            vr
        ))),
    }
}

pub fn into_xmm_q(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    vr: ValueRef,
    target: AsmRegisterXmm,
) -> Result<()> {
    match alloc.loc(vr) {
        Loc::Xmm(x) => {
            let src = xmm(x);
            if src != target {
                asm.movdqa(target, src).map_err(into_err)?;
            }
        }
        Loc::Spill(off) => asm
            .movdqu(target, xmmword_ptr(rbp - off))
            .map_err(into_err)?,
        Loc::Reg(_) | Loc::None => {
            return Err(Error::Backend(format!(
                "into_xmm_q from invalid loc ({:?})",
                vr
            )));
        }
    }
    Ok(())
}

pub fn working_xmm_for(
    alloc: &Allocation,
    dst_vr: ValueRef,
    scratch: AsmRegisterXmm,
) -> AsmRegisterXmm {
    match alloc.loc(dst_vr) {
        Loc::Xmm(x) => xmm(x),
        _ => scratch,
    }
}

fn into_err(e: iced_x86::IcedError) -> Error {
    Error::Backend(e.to_string())
}
