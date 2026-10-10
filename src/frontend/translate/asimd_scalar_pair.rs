//! Baseline integer ADDP Dd,Vn.2D. Source aliases are captured before the
//! scalar destination write, which clears the unused upper64 bits.
use disarm64::decoder::ASISDPAIR;
use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::bits;

pub fn translate(em: &mut IrEmitter<'_>, insn: ASISDPAIR) -> Result<InstStatus> {
    let raw = match insn {
        ASISDPAIR::ADDP_Sd_Vn(i) => i.0,
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };
    if bits(raw, 22, 2) != 3 { return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }); }
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let source = em.get_v_q(rn);
    let n = em.vec_extract_lo64(source);
    let m = em.vec_extract_hi64(source);
    let result = em.add(n, m, RegSize::X);
    em.set_v_d(rd, result);
    Ok(InstStatus::Continue)
}
