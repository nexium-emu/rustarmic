use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

// FCVTL/FCVTL2 single-to-double forms. The existing precision-conversion
// backend supplies default-FPCR values; this does not add FP16 or full FPSR
// exception/control emulation.
pub fn translate(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    if bits(raw, 22, 2) != 1 {
        return Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let first = if bit(raw, 30) == 0 { 0 } else { 2 };
    // Snapshot the whole source before either destination lane is written.
    let source = em.get_v_q(rn);
    let low = em.vec_extract_u32(source, first);
    let high = em.vec_extract_u32(source, first + 1);
    let low = em.push(Armlet::new(Op::FcvtSD, Ty::U64).with_args(&[low]));
    let high = em.push(Armlet::new(Op::FcvtSD, Ty::U64).with_args(&[high]));
    let result = em.vec_build_q(low, high);
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
