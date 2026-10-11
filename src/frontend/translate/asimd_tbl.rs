use disarm64::decoder::ASIMDTBL;

use crate::error::Result;
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

/// TBL / TBX with one to four table registers.
pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDTBL) -> Result<InstStatus> {
    use ASIMDTBL::*;
    let (raw, tbx) = match insn {
        TBL_Vd_LVn_Vm(i) => (i.0, false),
        TBX_Vd_LVn_Vm(i) => (i.0, true),
    };

    let q = bit(raw, 30) == 1;
    let len = bits(raw, 13, 2);
    let rm = bits(raw, 16, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    let t0 = em.get_v_q(rn);
    let indices = em.get_v_q(rm);
    // Out-of-range indices produce zero here (the full 128 bits are
    // computed; Q = 0 keeps the low half below).
    let looked_up = match len {
        0 => em.vec_tbl(t0, indices, true),
        1 => {
            let t1 = em.get_v_q((rn + 1) & 31);
            em.vec_tbl2(t0, t1, indices, true)
        }
        _ => {
            let t1 = em.get_v_q((rn + 1) & 31);
            let t2 = em.get_v_q((rn + 2) & 31);
            let first = em.vec_tbl3(t0, t1, t2, indices, true);
            if len == 2 {
                first
            } else {
                // Fourth register: indices 48..63, everything else (also
                // the ones that wrap below 48) is out of its range.
                let t3 = em.get_v_q((rn + 3) & 31);
                let c48 = em.const_u32(48);
                let v48 = em.vec_dup_gpr(c48, 0, true);
                let rebased = em.vec_sub(indices, v48, 0, true);
                let fourth = em.vec_tbl(t3, rebased, true);
                em.vec_orr(first, fourth, true)
            }
        }
    };
    let result = if tbx {
        // TBX keeps the destination byte for out-of-range indices.
        let limit = em.const_u32(16 * (len + 1));
        let vlimit = em.vec_dup_gpr(limit, 0, true);
        let out_of_range = em.vec_cmhs(indices, vlimit, 0, true);
        let old = em.get_v_q(rd);
        let kept = em.vec_and(old, out_of_range, true);
        em.vec_orr(looked_up, kept, true)
    } else {
        looked_up
    };
    if q {
        em.set_v_q(rd, result);
    } else {
        let lo = em.vec_extract_lo64(result);
        em.set_v_d(rd, lo);
    }
    Ok(InstStatus::Continue)
}
