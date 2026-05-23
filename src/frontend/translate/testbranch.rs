use disarm64::decoder::TESTBRANCH;

use crate::error::Result;
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Terminal, Ty};
use crate::util::bits::{bit, bits, sign_extend};

pub fn translate(em: &mut IrEmitter<'_>, insn: TESTBRANCH) -> Result<InstStatus> {
    use TESTBRANCH::*;
    let (raw, inverse) = match insn {
        TBZ_Rt_BIT_NUM_ADDR_PCREL14(i)  => (i.0, false),
        TBNZ_Rt_BIT_NUM_ADDR_PCREL14(i) => (i.0, true),
    };
    let b5    = bit(raw, 31);
    let b40   = bits(raw, 19, 5);
    let imm14 = bits(raw, 5, 14);
    let rt    = bits(raw, 0, 5) as u8;
    let bit_idx = ((b5 << 5) | b40) as u8;
    let offset = sign_extend(imm14 as u64, 14) << 2;
    let target = em.current_pc.wrapping_add(offset as u64);

    let val = em.get_x(rt);
    let op = if inverse { Op::TbNz } else { Op::TbZ };
    em.push(Armlet::new(op, Ty::Void)
        .with_args(&[val])
        .with_imm((target << 8) | (bit_idx as u64)));
    em.block.terminal = Terminal::TestBranchBit {
        value: val,
        bit: bit_idx,
        inverse,
        taken_pc: target,
        not_taken_pc: em.current_pc.wrapping_add(4),
    };
    Ok(InstStatus::Terminator)
}
