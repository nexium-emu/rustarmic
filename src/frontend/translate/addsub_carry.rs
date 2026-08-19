use disarm64::decoder::ADDSUB_CARRY;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: ADDSUB_CARRY) -> Result<InstStatus> {
    use ADDSUB_CARRY::*;
    let (raw, is_sub) = match insn {
        ADC_Rd_Rn_Rm(i) => (i.0, false),
        SBC_Rd_Rn_Rm(i) => (i.0, true),
        ADCS_Rd_Rn_Rm(_) | SBCS_Rd_Rn_Rm(_) => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };
    let sf = bit(raw, 31);
    let rm = bits(raw, 16, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let ty = if sf == 1 { Ty::U64 } else { Ty::U32 };

    let n = em.get_gpr(rn, size);
    let m = em.get_gpr(rm, size);
    let nzcv = em.get_nzcv();
    let op = match (is_sub, sf) {
        (false, 1) => Op::Adc64,
        (false, 0) => Op::Adc32,
        (true, 1) => Op::Sbc64,
        (true, 0) => Op::Sbc32,
        _ => unreachable!(),
    };
    let result = em.push(Armlet::new(op, ty).with_args(&[n, m, nzcv]));
    em.set_gpr(rd, result, size);
    Ok(InstStatus::Continue)
}
