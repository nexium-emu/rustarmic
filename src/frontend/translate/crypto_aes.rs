use disarm64::decoder::CRYPTOAES;

use crate::error::Result;
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::bits;

pub fn translate(em: &mut IrEmitter<'_>, insn: CRYPTOAES) -> Result<InstStatus> {
    use CRYPTOAES::*;
    let (raw, op) = match insn {
        AESE_Vd_Vn(i) => (i.0, Op::AesEncrypt),
        AESD_Vd_Vn(i) => (i.0, Op::AesDecrypt),
        AESMC_Vd_Vn(i) => (i.0, Op::AesMixColumns),
        AESIMC_Vd_Vn(i) => (i.0, Op::AesInverseMixColumns),
    };
    let rd = bits(raw, 0, 5) as u8;
    let source = em.get_v_q(bits(raw, 5, 5) as u8);
    let input = if matches!(op, Op::AesEncrypt | Op::AesDecrypt) {
        let state = em.get_v_q(rd);
        em.vec_eor(state, source, true)
    } else {
        source
    };
    let result = em.push(Armlet::new(op, Ty::U128).with_args(&[input]));
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
