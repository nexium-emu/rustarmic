use disarm64::decoder::CRYPTOSM4;

use crate::error::Result;
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::bits;

pub fn translate(em: &mut IrEmitter<'_>, insn: CRYPTOSM4) -> Result<InstStatus> {
    let (raw, key_schedule) = match insn {
        CRYPTOSM4::SM4E_Vd_Vn(i) => (i.0, false),
        CRYPTOSM4::SM4EKEY_Vd_Vn_Vm(i) => (i.0, true),
    };
    let rd = bits(raw, 0, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let state = em.get_v_q(if key_schedule { rn } else { rd });
    let keys = em.get_v_q(if key_schedule { bits(raw, 16, 5) as u8 } else { rn });
    let result = em.sm4_rounds(state, keys, key_schedule);
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
