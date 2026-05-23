use disarm64::decoder::IC_SYSTEM;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};

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
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    }
    Ok(InstStatus::Continue)
}
