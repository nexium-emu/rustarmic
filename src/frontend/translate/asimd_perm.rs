//! ASIMD permute group + EXT.
//!
//! Coverage so far:
//!   - EXT (byte-level concatenate-and-extract)
//!   - ZIP1 / ZIP2 (interleave low / high halves per lane size)
//! UZP1/UZP2/TRN1/TRN2 are unimplemented (they need pshufb masks; not
//! commonly hot, deferred for a follow-up).

use disarm64::decoder::{ASIMDEXT, ASIMDPERM};

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

pub fn translate_ext(em: &mut IrEmitter<'_>, insn: ASIMDEXT) -> Result<InstStatus> {
    use ASIMDEXT::*;
    let raw = match insn { EXT_Vd_Vn_Vm_IDX(i) => i.0 };

    let q     = bit(raw, 30) == 1;
    let rm    = bits(raw, 16, 5) as u8;
    let imm4  = bits(raw, 11, 4);
    let rn    = bits(raw, 5, 5) as u8;
    let rd    = bits(raw, 0, 5) as u8;
    // For the 64-bit (8B) form, only imm4[2:0] are valid.
    let byte_off = if q { imm4 } else { imm4 & 0x7 };

    let vn = em.get_v_q(rn);
    let vm = em.get_v_q(rm);
    let result = em.vec_ext(vn, vm, byte_off, q);
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}

pub fn translate_perm(em: &mut IrEmitter<'_>, insn: ASIMDPERM) -> Result<InstStatus> {
    use ASIMDPERM::*;
    let (raw, zip2) = match insn {
        ZIP1_Vd_Vn_Vm(i) => (i.0, false),
        ZIP2_Vd_Vn_Vm(i) => (i.0, true),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let q    = bit(raw, 30) == 1;
    let size = bits(raw, 22, 2); // 00=B, 01=H, 10=S, 11=D
    let rm   = bits(raw, 16, 5) as u8;
    let rn   = bits(raw, 5,  5) as u8;
    let rd   = bits(raw, 0,  5) as u8;

    let vn = em.get_v_q(rn);
    let vm = em.get_v_q(rm);
    let result = if zip2 { em.vec_zip2(vn, vm, size, q) } else { em.vec_zip1(vn, vm, size, q) };
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
