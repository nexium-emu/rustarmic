use disarm64::decoder::CONDSEL;

use crate::arch::RegSize;
use crate::error::Result;
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum Kind { Csel, Csinc, Csinv, Csneg }

pub fn translate(em: &mut IrEmitter<'_>, insn: CONDSEL) -> Result<InstStatus> {
    use CONDSEL::*;
    let (raw, kind) = match insn {
        CSEL_Rd_Rn_Rm_COND(i)  => (i.0, Kind::Csel),
        CSINC_Rd_Rn_Rm_COND(i) => (i.0, Kind::Csinc),
        CSINV_Rd_Rn_Rm_COND(i) => (i.0, Kind::Csinv),
        CSNEG_Rd_Rn_Rm_COND(i) => (i.0, Kind::Csneg),
    };

    let sf   = bit(raw, 31);
    let rm   = bits(raw, 16, 5) as u8;
    let cond = bits(raw, 12, 4);
    let rn   = bits(raw, 5, 5) as u8;
    let rd   = bits(raw, 0, 5) as u8;
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let ty   = if sf == 1 { Ty::U64 } else { Ty::U32 };

    let true_val = em.get_gpr(rn, size);
    let mut false_val = em.get_gpr(rm, size);
    let one = em.const_u64(1);
    let all_ones = em.const_u64(if sf == 1 { !0u64 } else { 0xFFFF_FFFFu64 });
    false_val = match kind {
        Kind::Csel  => false_val,
        Kind::Csinc => em.add(false_val, one, size),
        Kind::Csinv => em.eor(false_val, all_ones, size),
        Kind::Csneg => {
            let zero = em.const_u64(0);
            em.sub(zero, false_val, size)
        }
    };

    let nzcv = em.get_nzcv();
    let op = if sf == 1 { Op::Csel64 } else { Op::Csel32 };
    let result = em.push(Armlet::new(op, ty)
        .with_args(&[true_val, false_val, nzcv])
        .with_imm(cond as u64));

    em.set_x(rd, result);
    Ok(InstStatus::Continue)
}
