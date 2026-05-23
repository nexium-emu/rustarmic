//! Branches, exceptions, system. Top4 = 101x.

use crate::arch::Cond;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Terminal, Ty};
use crate::ir::block::ExceptionKind;
use crate::util::bits::{bit, bits, sign_extend};

pub fn translate(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let op0 = bits(inst, 29, 3);
    match op0 {
        0b000 | 0b100 => b_or_bl(em, inst),
        0b010         => cb_or_b_cond(em, inst),
        0b001 | 0b101 => cmp_test_branch(em, inst),
        0b110         => exception_or_system(em, inst),
        _ => Err(Error::Unsupported { pc: em.current_pc, opcode: inst }),
    }
}

fn b_or_bl(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let link = bit(inst, 31) == 1;
    let imm26 = bits(inst, 0, 26);
    let offset = sign_extend(imm26 as u64, 26) << 2;
    let target = em.current_pc.wrapping_add(offset as u64);
    em.branch(target, link);
    Ok(InstStatus::Terminator)
}

fn cb_or_b_cond(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    // Distinguish CBZ/CBNZ from B.cond using bit24.
    let bit24 = bit(inst, 24);
    if bit24 == 0 {
        // B.cond
        let cond = Cond::from_bits(bits(inst, 0, 4) as u8);
        let imm19 = bits(inst, 5, 19);
        let offset = sign_extend(imm19 as u64, 19) << 2;
        let target = em.current_pc.wrapping_add(offset as u64);
        em.branch_cond(cond, target);
        Ok(InstStatus::Terminator)
    } else {
        // CBZ/CBNZ
        let sf    = bit(inst, 31);
        let inv   = bit(inst, 24) == 1 && bit(inst, 23) == 0; // not actually right — handle in cmp_test_branch
        let _ = sf; let _ = inv;
        Err(Error::Unsupported { pc: em.current_pc, opcode: inst })
    }
}

fn cmp_test_branch(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let sf  = bit(inst, 31);
    let op_ = bit(inst, 24); // 0=CBZ/TBZ, 1=CBNZ/TBNZ
    let is_test = bits(inst, 29, 3) == 0b011 || bits(inst, 29, 3) == 0b111;
    let _ = sf;
    let _ = is_test;

    // Top3 == 001 → CBZ/CBNZ ; Top3 == 011 → TBZ/TBNZ
    let top3 = bits(inst, 29, 3);
    match top3 {
        0b001 | 0b101 => {
            // Compare and branch on (non-)zero
            let imm19 = bits(inst, 5, 19);
            let rt    = bits(inst, 0, 5) as u8;
            let offset = sign_extend(imm19 as u64, 19) << 2;
            let target = em.current_pc.wrapping_add(offset as u64);

            let val = if sf == 1 { em.get_x(rt) } else { em.get_w(rt) };
            let op = if op_ == 0 { Op::CbZ } else { Op::CbNz };
            em.push(Armlet::new(op, Ty::Void)
                .with_args(&[val])
                .with_imm(target));
            em.block.terminal = Terminal::CompareBranchZero {
                value: val,
                inverse: op_ == 1,
                taken_pc: target,
                not_taken_pc: em.current_pc.wrapping_add(4),
            };
            Ok(InstStatus::Terminator)
        }
        0b011 | 0b111 => {
            // Test bit and branch
            let b5   = bit(inst, 31);
            let b40  = bits(inst, 19, 5);
            let imm14 = bits(inst, 5, 14);
            let rt   = bits(inst, 0, 5) as u8;
            let bit_idx = ((b5 << 5) | b40) as u8;
            let offset = sign_extend(imm14 as u64, 14) << 2;
            let target = em.current_pc.wrapping_add(offset as u64);

            let val = em.get_x(rt);
            let op = if op_ == 0 { Op::TbZ } else { Op::TbNz };
            em.push(Armlet::new(op, Ty::Void)
                .with_args(&[val])
                .with_imm((target << 8) | (bit_idx as u64)));
            em.block.terminal = Terminal::TestBranchBit {
                value: val,
                bit: bit_idx,
                inverse: op_ == 1,
                taken_pc: target,
                not_taken_pc: em.current_pc.wrapping_add(4),
            };
            Ok(InstStatus::Terminator)
        }
        _ => Err(Error::Unsupported { pc: em.current_pc, opcode: inst }),
    }
}

fn exception_or_system(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    // Exception-generating encoding: bits 31..24 = 1101_0100 (= 0xD4),
    // i.e. bits(inst, 24, 8) == 0b1101_0100.
    if bits(inst, 24, 8) == 0b1101_0100 {
        let ll = bits(inst, 0, 2);
        let opc = bits(inst, 21, 3);
        let imm16 = bits(inst, 5, 16);
        return match (opc, ll) {
            (0b000, 0b01) => {
                em.push(Armlet::new(Op::Svc, Ty::Void).with_imm(imm16 as u64));
                em.block.terminal = Terminal::Exception { kind: ExceptionKind::Svc, imm: imm16 };
                Ok(InstStatus::Terminator)
            }
            (0b000, 0b10) => {
                em.push(Armlet::new(Op::Hvc, Ty::Void).with_imm(imm16 as u64));
                em.block.terminal = Terminal::Exception { kind: ExceptionKind::Hvc, imm: imm16 };
                Ok(InstStatus::Terminator)
            }
            (0b001, 0b00) => {
                em.push(Armlet::new(Op::Brk, Ty::Void).with_imm(imm16 as u64));
                em.block.terminal = Terminal::Exception { kind: ExceptionKind::Brk, imm: imm16 };
                Ok(InstStatus::Terminator)
            }
            _ => Err(Error::Unsupported { pc: em.current_pc, opcode: inst }),
        };
    }
    if bits(inst, 25, 7) == 0b1101011 {
        // Unconditional branch (register): BR/BLR/RET
        let opc = bits(inst, 21, 4);
        let rn  = bits(inst, 5, 5) as u8;
        let target = em.get_x(rn);
        return match opc {
            0b0000 => { em.branch_indirect(target, false, false); Ok(InstStatus::Terminator) }
            0b0001 => { em.branch_indirect(target, true,  false); Ok(InstStatus::Terminator) }
            0b0010 => { em.branch_indirect(target, false, true);  Ok(InstStatus::Terminator) }
            _ => Err(Error::Unsupported { pc: em.current_pc, opcode: inst }),
        };
    }
    if bits(inst, 22, 10) == 0b1101010100 {
        // System / Hint / barrier.
        let op0 = bit(inst, 21);
        if op0 == 0 && bits(inst, 12, 4) == 0b0010 && bits(inst, 16, 3) == 0b011 {
            let crm = bits(inst, 8, 4);
            let op2 = bits(inst, 5, 3);
            let hint_code = (crm << 3) | op2;
            em.push(Armlet::new(Op::Hint, Ty::Void).with_imm(hint_code as u64));
            return Ok(InstStatus::Continue);
        }
        return Err(Error::Unsupported { pc: em.current_pc, opcode: inst });
    }
    Err(Error::Unsupported { pc: em.current_pc, opcode: inst })
}
