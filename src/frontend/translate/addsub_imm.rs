use disarm64::decoder::ADDSUB_IMM;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: ADDSUB_IMM) -> Result<InstStatus> {
    use ADDSUB_IMM::*;
    let (raw, sub, set_flags) = match insn {
        ADD_Rd_SP_Rn_SP_AIMM(i) => (i.0, false, false),
        ADDS_Rd_Rn_SP_AIMM(i) => (i.0, false, true),
        SUB_Rd_SP_Rn_SP_AIMM(i) => (i.0, true, false),
        SUBS_Rd_Rn_SP_AIMM(i) => (i.0, true, true),
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };

    let sf = bit(raw, 31);
    let sh = bit(raw, 22);
    let imm12 = bits(raw, 10, 12);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let mut imm = imm12 as u64;
    if sh == 1 {
        imm <<= 12;
    }
    if sf == 0 {
        imm &= 0xFFFF_FFFF;
    }

    let sp_form = !set_flags;
    let a = em.get_x_or_sp(rn, sp_form);
    let b = em.const_u64(imm);

    if set_flags {
        let result = if sub {
            em.subs(a, b, size)
        } else {
            em.adds(a, b, size)
        };
        em.set_gpr(rd, result, size);
    } else {
        let result = if sub {
            em.sub(a, b, size)
        } else {
            em.add(a, b, size)
        };
        em.set_x_or_sp(rd, result, sp_form);
    }
    Ok(InstStatus::Continue)
}
