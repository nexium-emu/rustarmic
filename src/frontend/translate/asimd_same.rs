//! ASIMD "same"-form three-operand vector ops (Vd, Vn, Vm with matching shapes).
//!
//! Coverage so far: the bitwise logical ops (AND/ORR/EOR/BIC/ORN), which gives
//! us the `MOV Vd.16B, Vn.16B` alias for free (encoded as `ORR Vd, Vn, Vn`).
//! The arithmetic per-lane ops (ADD/SUB/MUL/...) and FP per-lane ops come in
//! the next phases.

use disarm64::decoder::ASIMDSAME;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum Logic { And, Orr, Eor, Bic, Orn }

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDSAME) -> Result<InstStatus> {
    use ASIMDSAME::*;
    // ASIMDSAME logical ops share a layout: bit 30 = Q (128-bit form when 1),
    // bits 23:22 = size (always 0 for logicals), bits 22:23 select op via
    // higher decode. We just need to know which logic op to perform.
    let (raw, logic) = match insn {
        AND_Vd_Vn_Vm(i) => (i.0, Logic::And),
        ORR_Vd_Vn_Vm(i) => (i.0, Logic::Orr),
        EOR_Vd_Vn_Vm(i) => (i.0, Logic::Eor),
        BIC_Vd_Vn_Vm(i) => (i.0, Logic::Bic),
        ORN_Vd_Vn_Vm(i) => (i.0, Logic::Orn),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let q  = bit(raw, 30);
    let rm = bits(raw, 16, 5) as u8;
    let rn = bits(raw, 5,  5) as u8;
    let rd = bits(raw, 0,  5) as u8;

    // Logical ops are bitwise — they don't care about lane shape, so we can
    // operate on the two u64 halves directly and combine.
    let vn_q = em.get_v_q(rn);
    let vm_q = em.get_v_q(rm);
    let vn_lo = em.vec_extract_lo64(vn_q);
    let vn_hi = em.vec_extract_hi64(vn_q);
    let vm_lo = em.vec_extract_lo64(vm_q);
    let vm_hi = em.vec_extract_hi64(vm_q);

    let (r_lo, r_hi) = match logic {
        Logic::And => (em.and(vn_lo, vm_lo, RegSize::X), em.and(vn_hi, vm_hi, RegSize::X)),
        Logic::Orr => (em.or (vn_lo, vm_lo, RegSize::X), em.or (vn_hi, vm_hi, RegSize::X)),
        Logic::Eor => (em.eor(vn_lo, vm_lo, RegSize::X), em.eor(vn_hi, vm_hi, RegSize::X)),
        Logic::Bic => {
            // BIC = Vn AND NOT(Vm). Materialise the NOT as XOR with all-ones.
            let ones = em.const_u64(u64::MAX);
            let nm_lo = em.eor(vm_lo, ones, RegSize::X);
            let ones2 = em.const_u64(u64::MAX);
            let nm_hi = em.eor(vm_hi, ones2, RegSize::X);
            (em.and(vn_lo, nm_lo, RegSize::X), em.and(vn_hi, nm_hi, RegSize::X))
        }
        Logic::Orn => {
            let ones = em.const_u64(u64::MAX);
            let nm_lo = em.eor(vm_lo, ones, RegSize::X);
            let ones2 = em.const_u64(u64::MAX);
            let nm_hi = em.eor(vm_hi, ones2, RegSize::X);
            (em.or(vn_lo, nm_lo, RegSize::X), em.or(vn_hi, nm_hi, RegSize::X))
        }
    };

    let (final_lo, final_hi) = if q == 1 {
        (r_lo, r_hi)
    } else {
        // 8B/D form: clear upper 64 bits.
        let zero = em.const_u64(0);
        (r_lo, zero)
    };

    let result = em.vec_build_q(final_lo, final_hi);
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
