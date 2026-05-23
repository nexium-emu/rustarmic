use disarm64::decoder::DP_2SRC;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

enum Kind { Lslv, Lsrv, Asrv, Rorv, Udiv, Sdiv }

pub fn translate(em: &mut IrEmitter<'_>, insn: DP_2SRC) -> Result<InstStatus> {
    use DP_2SRC::*;
    let (raw, kind) = match insn {
        LSLV_Rd_Rn_Rm(i) => (i.0, Kind::Lslv),
        LSRV_Rd_Rn_Rm(i) => (i.0, Kind::Lsrv),
        ASRV_Rd_Rn_Rm(i) => (i.0, Kind::Asrv),
        RORV_Rd_Rn_Rm(i) => (i.0, Kind::Rorv),
        UDIV_Rd_Rn_Rm(i) => (i.0, Kind::Udiv),
        SDIV_Rd_Rn_Rm(i) => (i.0, Kind::Sdiv),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let sf = bit(raw, 31);
    let rm = bits(raw, 16, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let ty   = if sf == 1 { Ty::U64 } else { Ty::U32 };

    let n = em.get_gpr(rn, size);
    let m = em.get_gpr(rm, size);

    let result = match kind {
        Kind::Lslv => em.lsl(n, m, size),
        Kind::Lsrv => em.lsr(n, m, size),
        Kind::Asrv => em.asr(n, m, size),
        Kind::Rorv => em.ror(n, m, size),
        Kind::Udiv => {
            let op = if sf == 1 { Op::UDiv64 } else { Op::UDiv32 };
            em.push(Armlet::new(op, ty).with_args(&[n, m]))
        }
        Kind::Sdiv => {
            let op = if sf == 1 { Op::SDiv64 } else { Op::SDiv32 };
            em.push(Armlet::new(op, ty).with_args(&[n, m]))
        }
    };
    em.set_gpr(rd, result, size);
    Ok(InstStatus::Continue)
}
