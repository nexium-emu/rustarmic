use disarm64::decoder::COMPBRANCH;

use crate::arch::RegSize;
use crate::error::Result;
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Terminal, Ty};
use crate::util::bits::{bit, bits, sign_extend};

pub fn translate(em: &mut IrEmitter<'_>, insn: COMPBRANCH) -> Result<InstStatus> {
    use COMPBRANCH::*;
    let (raw, inverse) = match insn {
        CBZ_Rt_ADDR_PCREL19(i)  => (i.0, false),
        CBNZ_Rt_ADDR_PCREL19(i) => (i.0, true),
    };
    let sf    = bit(raw, 31);
    let imm19 = bits(raw, 5, 19);
    let rt    = bits(raw, 0, 5) as u8;
    let offset = sign_extend(imm19 as u64, 19) << 2;
    let target = em.current_pc.wrapping_add(offset as u64);

    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let val = em.get_gpr(rt, size);

    let op = if inverse { Op::CbNz } else { Op::CbZ };
    em.push(Armlet::new(op, Ty::Void).with_args(&[val]).with_imm(target));
    em.block.terminal = Terminal::CompareBranchZero {
        value: val,
        inverse,
        taken_pc: target,
        not_taken_pc: em.current_pc.wrapping_add(4),
    };
    Ok(InstStatus::Terminator)
}
