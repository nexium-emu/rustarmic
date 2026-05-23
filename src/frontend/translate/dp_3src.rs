use disarm64::decoder::DP_3SRC;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum Kind { Madd, Msub, Smaddl, Umaddl, Smsubl, Umsubl }

pub fn translate(em: &mut IrEmitter<'_>, insn: DP_3SRC) -> Result<InstStatus> {
    use DP_3SRC::*;
    let (raw, kind) = match insn {
        MADD_Rd_Rn_Rm_Ra(i)   => (i.0, Kind::Madd),
        MSUB_Rd_Rn_Rm_Ra(i)   => (i.0, Kind::Msub),
        SMADDL_Rd_Rn_Rm_Ra(i) => (i.0, Kind::Smaddl),
        UMADDL_Rd_Rn_Rm_Ra(i) => (i.0, Kind::Umaddl),
        SMSUBL_Rd_Rn_Rm_Ra(i) => (i.0, Kind::Smsubl),
        UMSUBL_Rd_Rn_Rm_Ra(i) => (i.0, Kind::Umsubl),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let sf = bit(raw, 31);
    let rm = bits(raw, 16, 5) as u8;
    let ra = bits(raw, 10, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    match kind {
        Kind::Madd | Kind::Msub => {
            let size = if sf == 1 { RegSize::X } else { RegSize::W };
            let n = em.get_gpr(rn, size);
            let m = em.get_gpr(rm, size);
            let a = em.get_gpr(ra, size);
            let prod = em.push(crate::ir::Armlet::new(
                if sf == 1 { crate::ir::Op::Mul64 } else { crate::ir::Op::Mul32 },
                if sf == 1 { crate::ir::Ty::U64 } else { crate::ir::Ty::U32 },
            ).with_args(&[n, m]));
            let result = match kind {
                Kind::Madd => em.add(a, prod, size),
                Kind::Msub => em.sub(a, prod, size),
                _ => unreachable!(),
            };
            em.set_gpr(rd, result, size);
        }
        Kind::Smaddl | Kind::Umaddl | Kind::Smsubl | Kind::Umsubl => {
            let n = em.get_w(rn);
            let m = em.get_w(rm);
            let a = em.get_x(ra);
            let signed = matches!(kind, Kind::Smaddl | Kind::Smsubl);

            let n64 = widen_w_to_x(em, n, signed);
            let m64 = widen_w_to_x(em, m, signed);
            let prod = em.push(crate::ir::Armlet::new(crate::ir::Op::Mul64, crate::ir::Ty::U64)
                .with_args(&[n64, m64]));
            let result = match kind {
                Kind::Smaddl | Kind::Umaddl => em.add(a, prod, RegSize::X),
                Kind::Smsubl | Kind::Umsubl => em.sub(a, prod, RegSize::X),
                _ => unreachable!(),
            };
            em.set_x(rd, result);
        }
    }
    Ok(InstStatus::Continue)
}

fn widen_w_to_x(em: &mut IrEmitter<'_>, v: crate::ir::ValueRef, signed: bool) -> crate::ir::ValueRef {
    let mask = em.const_u64(0xFFFF_FFFF);
    let masked = em.and(v, mask, RegSize::X);
    if signed {
        let shl = em.const_u64(32);
        let s1 = em.lsl(masked, shl, RegSize::X);
        let shl2 = em.const_u64(32);
        em.asr(s1, shl2, RegSize::X)
    } else {
        masked
    }
}
