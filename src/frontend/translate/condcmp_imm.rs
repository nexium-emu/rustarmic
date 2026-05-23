use disarm64::decoder::CONDCMP_IMM;

use crate::arch::RegSize;
use crate::error::Result;
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: CONDCMP_IMM) -> Result<InstStatus> {
    use CONDCMP_IMM::*;
    let (raw, is_sub) = match insn {
        CCMP_Rn_CCMP_IMM_NZCV_COND(i) => (i.0, true),
        CCMN_Rn_CCMP_IMM_NZCV_COND(i) => (i.0, false),
    };
    let sf       = bit(raw, 31);
    let imm5     = bits(raw, 16, 5);
    let cond     = bits(raw, 12, 4);
    let rn       = bits(raw, 5, 5) as u8;
    let nzcv_imm = bits(raw, 0, 4);

    let size = if sf == 1 { RegSize::X } else { RegSize::W };

    let old_nzcv = em.get_nzcv();
    let a = em.get_gpr(rn, size);
    let b = em.const_u64(imm5 as u64);

    if is_sub { em.subs(a, b, size); } else { em.adds(a, b, size); }

    let compare_nzcv = em.get_nzcv();
    let imm_nzcv = em.push(Armlet::new(Op::ConstU32, Ty::U32).with_imm(nzcv_imm as u64));

    let selected = em.push(Armlet::new(Op::Csel32, Ty::U32)
        .with_args(&[compare_nzcv, imm_nzcv, old_nzcv])
        .with_imm(cond as u64));
    em.set_nzcv(selected);
    Ok(InstStatus::Continue)
}
