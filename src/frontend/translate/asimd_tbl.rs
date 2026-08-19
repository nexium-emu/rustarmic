use disarm64::decoder::ASIMDTBL;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDTBL) -> Result<InstStatus> {
    use ASIMDTBL::*;
    let raw = match insn {
        TBL_Vd_LVn_Vm(i) => i.0,
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };

    let q = bit(raw, 30) == 1;
    let len = bits(raw, 13, 2);
    let rm = bits(raw, 16, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    let t0 = em.get_v_q(rn);
    let indices = em.get_v_q(rm);
    let result = match len {
        0 => em.vec_tbl(t0, indices, q),
        1 => {
            let t1 = em.get_v_q((rn + 1) & 31);
            em.vec_tbl2(t0, t1, indices, q)
        }
        2 => {
            let t1 = em.get_v_q((rn + 1) & 31);
            let t2 = em.get_v_q((rn + 2) & 31);
            em.vec_tbl3(t0, t1, t2, indices, q)
        }
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: raw,
            });
        }
    };
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
