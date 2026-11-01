use disarm64::decoder::LDST_IMM10;

use crate::arch::RegSize;
use crate::error::Result;
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits, sign_extend};

pub fn translate(em: &mut IrEmitter<'_>, insn: LDST_IMM10) -> Result<InstStatus> {
    let raw = match insn {
        LDST_IMM10::LDRAA_Rt_ADDR_SIMM10(i) => i.0,
        LDST_IMM10::LDRAB_Rt_ADDR_SIMM10(i) => i.0,
    };
    let imm10 = (bit(raw, 22) << 9) | bits(raw, 12, 9);
    let offset = (sign_extend(imm10 as u64, 10) as u64) << 3;
    let rn = bits(raw, 5, 5) as u8;
    let rt = bits(raw, 0, 5) as u8;
    let base = em.get_x_or_sp(rn, true);
    let delta = em.const_u64(offset);
    let address = em.add(base, delta, RegSize::X);
    let value = em.load(address, 8);
    if bit(raw, 11) == 1 {
        em.set_x_or_sp(rn, address, true);
    }
    em.set_x(rt, value);
    Ok(InstStatus::Continue)
}
