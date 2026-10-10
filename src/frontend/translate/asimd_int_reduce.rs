use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

/// Integer across-lane min/max: 8B/16B, 4H/8H, and 4S.
pub fn translate(
    em: &mut IrEmitter<'_>,
    raw: u32,
    signed: bool,
    maximum: bool,
) -> Result<InstStatus> {
    let q = bit(raw, 30) != 0;
    let size = bits(raw, 22, 2);
    if size == 3 || (size == 2 && !q) {
        return Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    // Capture both source halves before an aliased scalar destination clears Q.
    let source = em.get_v_q(rn);
    let low = em.vec_extract_lo64(source);
    let high = em.vec_extract_hi64(source);
    let mode = u64::from(signed)
        | (u64::from(maximum) << 1)
        | ((size as u64) << 2)
        | (u64::from(q) << 4);
    let result = em.push(
        Armlet::new(Op::IntMinMaxReduce32, Ty::U32)
            .with_args(&[low, high])
            .with_imm(mode),
    );
    // The helper masks B/H results rather than sign-extending into unused bits.
    em.set_v_s(rd, result);
    Ok(InstStatus::Continue)
}
