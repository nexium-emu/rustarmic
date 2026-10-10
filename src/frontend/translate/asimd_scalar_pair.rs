//! Baseline integer ADDP Dd,Vn.2D. Source aliases are captured before the
//! scalar destination write, which clears the unused upper64 bits.
use disarm64::decoder::ASISDPAIR;
use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::bits;

pub fn translate(em: &mut IrEmitter<'_>, insn: ASISDPAIR) -> Result<InstStatus> {
    let (raw, mode) = match insn {
        ASISDPAIR::ADDP_Sd_Vn(i) => (i.0, 5),
        ASISDPAIR::FADDP_Sd_S_S_Vn_V_2S(i) => (i.0, 0),
        ASISDPAIR::FMAXP_Sd_S_S_Vn_V_2S(i) => (i.0, 1),
        ASISDPAIR::FMINP_Sd_S_S_Vn_V_2S(i) => (i.0, 2),
        ASISDPAIR::FMAXNMP_Sd_S_S_Vn_V_2S(i) => (i.0, 3),
        ASISDPAIR::FMINNMP_Sd_S_S_Vn_V_2S(i) => (i.0, 4),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };
    if mode == 5 && bits(raw, 22, 2) != 3 { return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }); }
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let source = em.get_v_q(rn);
    if mode == 5 {
        let n = em.vec_extract_lo64(source);
        let m = em.vec_extract_hi64(source);
        let result = em.add(n, m, RegSize::X);
        em.set_v_d(rd, result);
    } else {
        let double = bits(raw, 22, 1) != 0;
        let (n, m, op, ty) = if double {
            (em.vec_extract_lo64(source), em.vec_extract_hi64(source), Op::ScalarFpPair64, Ty::U64)
        } else {
            (em.vec_extract_u32(source, 0), em.vec_extract_u32(source, 1), Op::ScalarFpPair32, Ty::U32)
        };
        let result = em.push(Armlet::new(op, ty).with_args(&[n, m]).with_imm(mode));
        if double { em.set_v_d(rd, result); } else { em.set_v_s(rd, result); }
    }
    Ok(InstStatus::Continue)
}
