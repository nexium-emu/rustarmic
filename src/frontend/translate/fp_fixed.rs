use disarm64::decoder::FLOAT2FIX;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: FLOAT2FIX) -> Result<InstStatus> {
    use FLOAT2FIX::*;
    let raw = match insn {
        FCVTZS_Rd_Fn_FBITS(i) => i.0,
        FCVTZS_Rd_W_Fn_S_D_FBITS_imm_1_32(i) => i.0,
        FCVTZU_Rd_Fn_FBITS(i) => i.0,
        FCVTZU_Rd_W_Fn_S_D_FBITS_imm_1_32(i) => i.0,
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            })
        }
    };
    let scale = bits(raw, 10, 6);
    if bit(raw, 31) == 0 && scale < 32 {
        return Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    super::fp_conv::translate_float_to_int(em, raw, 64 - scale)
}
