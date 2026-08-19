use disarm64::decoder::EXCEPTION;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::block::ExceptionKind;
use crate::ir::{Armlet, IrEmitter, Op, Terminal, Ty};
use crate::util::bits::bits;

pub fn translate(em: &mut IrEmitter<'_>, insn: EXCEPTION) -> Result<InstStatus> {
    use EXCEPTION::*;
    let (op, kind, raw) = match insn {
        BRK_EXCEPTION(i) => (Op::Brk, ExceptionKind::Brk, i.0),
        SVC_EXCEPTION(i) => (Op::Svc, ExceptionKind::Svc, i.0),
        HVC_EXCEPTION(i) => (Op::Hvc, ExceptionKind::Hvc, i.0),
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };
    let imm16 = bits(raw, 5, 16);
    em.push(Armlet::new(op, Ty::Void).with_imm(imm16 as u64));
    em.block.terminal = Terminal::Exception { kind, imm: imm16 };
    Ok(InstStatus::Terminator)
}
