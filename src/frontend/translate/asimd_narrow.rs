use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

// Kind0: signed→signed,1:unsigned→unsigned,2:signed→unsigned.
pub fn translate(em: &mut IrEmitter<'_>, raw: u32, scalar: bool, kind: u32) -> Result<InstStatus> {
    let size = bits(raw, 22, 2);
    if size == 3 {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let upper = !scalar && bit(raw, 30) != 0;
    // Both the full source and narrow2's old low64 must be captured before the
    // helper/write, including Rd==Rn and V31. Only scalar lane0 is evaluated.
    let source = em.get_v_q(rn);
    let low = em.vec_extract_lo64(source);
    let high = em.vec_extract_hi64(source);
    let previous = upper.then(|| em.get_v_d(rd));
    let mode = size as u64 | ((kind as u64) << 2) | ((scalar as u64) << 4);
    let result = em.push(
        Armlet::new(Op::SaturatingNarrow, Ty::U64)
            .with_args(&[low, high])
            .with_imm(mode),
    );
    if let Some(previous) = previous {
        let vector = em.vec_build_q(previous, result);
        em.set_v_q(rd, vector);
    } else {
        // The scalar result is zero-extended to64; vector first fills64.
        // Either form clears every unused upper bit of the128-bit register.
        em.set_v_d(rd, result);
    }
    Ok(InstStatus::Continue)
}
