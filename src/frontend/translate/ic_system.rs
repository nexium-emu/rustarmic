use disarm64::decoder::IC_SYSTEM;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: IC_SYSTEM) -> Result<InstStatus> {
    use IC_SYSTEM::*;
    match insn {
        CLREX_UIMM4(_) => {
            em.push(Armlet::new(Op::Clrex, Ty::Void));
        }
        HINT_UIMM7(_) | DGH(_) | SB(_) => {
            em.push(Armlet::new(Op::Hint, Ty::Void));
        }
        DMB_BARRIER(_) | DSB_BARRIER(_) | DSB_BARRIER_DSB_NXS(_) | ISB_BARRIER_ISB(_) => {
            em.push(Armlet::new(Op::MemoryBarrier, Ty::Void));
        }
        MSR_PSTATEFIELD_UIMM4(_) => {
            em.push(Armlet::new(Op::Hint, Ty::Void));
        }
        MRS_Rt_SYSREG(i) => {
            let raw = i.0;
            let sysreg = encode_sysreg(raw);
            let rt = bits(raw, 0, 5) as u8;
            let val = em.push(Armlet::new(Op::Mrs, Ty::U64).with_imm(sysreg as u64));
            em.set_gpr(rt, val, RegSize::X);
        }
        MSR_SYSREG_Rt(i) => {
            let raw = i.0;
            let sysreg = encode_sysreg(raw);
            let rt = bits(raw, 0, 5) as u8;
            let val = em.get_gpr(rt, RegSize::X);
            em.push(Armlet::new(Op::Msr, Ty::Void).with_args(&[val]).with_imm(sysreg as u64));
        }
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    }
    Ok(InstStatus::Continue)
}

fn encode_sysreg(raw: u32) -> u16 {
    let op0 = 2 | bit(raw, 19);
    let op1 = bits(raw, 16, 3);
    let crn = bits(raw, 12, 4);
    let crm = bits(raw, 8, 4);
    let op2 = bits(raw, 5, 3);
    crate::arch::sysreg::pack(op0, op1, crn, crm, op2)
}
