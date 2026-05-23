//! Data processing — register (top4 = x101).
//!
//! Subset implemented up front: shifted-register ADD/SUB/AND/ORR/EOR/BIC and
//! the logical variants of MOV (which are just `ORR XZR, Rm`), plus
//! MUL/MADD/MSUB and conditional select.

use crate::arch::{Cond, RegSize};
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, ArmletFlags, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    // op0 (28)/op1 (24)/op2 (21..23) decode tree.
    let op1 = bit(inst, 28);
    if op1 == 0 {
        // Logical (shifted register) / Add-sub (shifted/extended register)
        let op2 = bit(inst, 24);
        if op2 == 0 {
            // Logical (shifted register) if bit24==0 && bit21 free
            logical_shifted_reg(em, inst)
        } else {
            // Add/sub shifted or extended register
            let bit21 = bit(inst, 21);
            if bit21 == 0 {
                add_sub_shifted_reg(em, inst)
            } else {
                add_sub_extended_reg(em, inst)
            }
        }
    } else {
        // Data-processing (3-source) / conditional select / data-proc (1/2 source)
        let op2 = bits(inst, 21, 3);
        if bits(inst, 24, 4) == 0b1010 && op2 == 0b100 {
            cond_select(em, inst)
        } else if bits(inst, 24, 4) == 0b1011 && bit(inst, 21) == 1 {
            data_proc_3_source(em, inst)
        } else if bits(inst, 24, 4) == 0b1010 && bits(inst, 21, 3) == 0b000 {
            // ADC / SBC
            adc_sbc(em, inst)
        } else {
            Err(Error::Unsupported { pc: em.current_pc, opcode: inst })
        }
    }
}

/// AND/ORR/EOR/BIC/ORN/EON/ANDS — shifted register. Also subsumes MOV (= ORR XZR, Rm).
fn logical_shifted_reg(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let sf    = bit(inst, 31);
    let opc   = bits(inst, 29, 2);
    let shift = bits(inst, 22, 2);
    let n_bit = bit(inst, 21);
    let rm    = bits(inst, 16, 5) as u8;
    let imm6  = bits(inst, 10, 6);
    let rn    = bits(inst, 5, 5) as u8;
    let rd    = bits(inst, 0, 5) as u8;

    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let width = if sf == 1 { 64 } else { 32 };
    if sf == 0 && imm6 >= 32 {
        return Err(Error::Decode { pc: em.current_pc, opcode: inst });
    }

    let a = em.get_gpr(rn, size);
    let mut b = em.get_gpr(rm, size);

    // Apply shift
    if imm6 != 0 {
        let amt = em.const_u64(imm6 as u64);
        b = match shift {
            0b00 => em.lsl(b, amt, size),
            0b01 => em.lsr(b, amt, size),
            0b10 => em.asr(b, amt, size),
            0b11 => em.ror(b, amt, size),
            _ => unreachable!(),
        };
        let _ = width;
    } else if shift == 0b11 {
        // ROR by 0 is a no-op for ROR, but the encoding is still permitted.
    }

    // Invert b for BIC/ORN/EON
    if n_bit == 1 {
        let all_ones = em.const_u64(if sf == 1 { !0u64 } else { 0xFFFF_FFFFu64 });
        b = em.eor(b, all_ones, size);
    }

    let result = match opc {
        0b00 | 0b11 => em.and(a, b, size),
        0b01        => em.or(a, b, size),
        0b10        => em.eor(a, b, size),
        _ => unreachable!(),
    };

    if opc == 0b11 {
        // ANDS sets flags from result.
        let zero = em.const_u64(0);
        let (_, flag) = em.subs(result, zero, size);
        em.set_nzcv(flag);
    }

    em.set_x(rd, result);
    Ok(InstStatus::Continue)
}

/// ADD/SUB (shifted register), with optional flag setting.
fn add_sub_shifted_reg(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let sf    = bit(inst, 31);
    let op_   = bit(inst, 30); // 0=ADD, 1=SUB
    let s     = bit(inst, 29);
    let shift = bits(inst, 22, 2);
    let rm    = bits(inst, 16, 5) as u8;
    let imm6  = bits(inst, 10, 6);
    let rn    = bits(inst, 5, 5) as u8;
    let rd    = bits(inst, 0, 5) as u8;

    if shift == 0b11 {
        return Err(Error::Decode { pc: em.current_pc, opcode: inst });
    }
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    if sf == 0 && imm6 >= 32 {
        return Err(Error::Decode { pc: em.current_pc, opcode: inst });
    }

    let a = em.get_gpr(rn, size);
    let mut b = em.get_gpr(rm, size);
    if imm6 != 0 {
        let amt = em.const_u64(imm6 as u64);
        b = match shift {
            0b00 => em.lsl(b, amt, size),
            0b01 => em.lsr(b, amt, size),
            0b10 => em.asr(b, amt, size),
            _ => unreachable!(),
        };
    }

    if s == 1 {
        let (result, flag) = if op_ == 0 { em.adds(a, b, size) } else { em.subs(a, b, size) };
        em.set_nzcv(flag);
        em.set_x(rd, result);
    } else {
        let result = if op_ == 0 { em.add(a, b, size) } else { em.sub(a, b, size) };
        em.set_x(rd, result);
    }
    Ok(InstStatus::Continue)
}

/// ADD/SUB (extended register).
fn add_sub_extended_reg(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let sf      = bit(inst, 31);
    let op_     = bit(inst, 30);
    let s       = bit(inst, 29);
    let rm      = bits(inst, 16, 5) as u8;
    let option_ = bits(inst, 13, 3);
    let imm3    = bits(inst, 10, 3);
    let rn      = bits(inst, 5, 5) as u8;
    let rd      = bits(inst, 0, 5) as u8;

    if imm3 > 4 {
        return Err(Error::Decode { pc: em.current_pc, opcode: inst });
    }
    let size = if sf == 1 { RegSize::X } else { RegSize::W };

    let sp_form = s == 0;
    let a = em.get_x_or_sp(rn, sp_form);

    // Extend Rm per option_
    let mut b = em.get_x(rm);
    let (extracted_width, signed) = match option_ {
        0b000 => (8, false),   // UXTB
        0b001 => (16, false),  // UXTH
        0b010 => (32, false),  // UXTW
        0b011 => (64, false),  // UXTX (or LSL when option == 011 in 64-bit form)
        0b100 => (8, true),    // SXTB
        0b101 => (16, true),   // SXTH
        0b110 => (32, true),   // SXTW
        0b111 => (64, true),   // SXTX
        _ => unreachable!(),
    };

    if extracted_width < 64 {
        let mask = (1u64 << extracted_width) - 1;
        let mask_c = em.const_u64(mask);
        b = em.and(b, mask_c, RegSize::X);
        if signed {
            // Sign-extend by left-shift then ASR
            let shl = em.const_u64((64 - extracted_width) as u64);
            let sh1 = em.lsl(b, shl, RegSize::X);
            let shl2 = em.const_u64((64 - extracted_width) as u64);
            b = em.asr(sh1, shl2, RegSize::X);
        }
    }
    if imm3 != 0 {
        let amt = em.const_u64(imm3 as u64);
        b = em.lsl(b, amt, RegSize::X);
    }

    // Truncate to operand size for the actual ALU op
    if size == RegSize::W {
        let mask_c = em.const_u64(0xFFFF_FFFF);
        b = em.and(b, mask_c, RegSize::X);
    }

    if s == 1 {
        let (result, flag) = if op_ == 0 { em.adds(a, b, size) } else { em.subs(a, b, size) };
        em.set_nzcv(flag);
        em.set_x(rd, result);
    } else {
        let result = if op_ == 0 { em.add(a, b, size) } else { em.sub(a, b, size) };
        em.set_x_or_sp(rd, result, sp_form);
    }
    Ok(InstStatus::Continue)
}

/// ADC/SBC and the ADCS/SBCS variants.
fn adc_sbc(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let sf  = bit(inst, 31);
    let op_ = bit(inst, 30); // 0=ADC, 1=SBC
    let s   = bit(inst, 29);
    let rm  = bits(inst, 16, 5) as u8;
    let rn  = bits(inst, 5, 5) as u8;
    let rd  = bits(inst, 0, 5) as u8;
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let ty   = if sf == 1 { Ty::U64 } else { Ty::U32 };

    let lhs = em.get_gpr(rn, size);
    let rhs = em.get_gpr(rm, size);
    let carry = em.get_nzcv(); // backend extracts C bit

    let op = match (op_, sf) {
        (0, 1) => Op::Adc64,
        (0, 0) => Op::Adc32,
        (1, 1) => Op::Sbc64,
        (1, 0) => Op::Sbc32,
        _ => unreachable!(),
    };
    let mut a = Armlet::new(op, ty);
    a = a.with_args(&[lhs, rhs, carry]);
    if s == 1 { a = a.with_flags(ArmletFlags::NZCV_LIVE); }
    let result = em.push(a);

    if s == 1 {
        let flag = em.push(Armlet::new(Op::Identity, Ty::Nzcv)
            .with_args(&[result])
            .with_flags(ArmletFlags::NZCV_LIVE));
        em.set_nzcv(flag);
    }
    em.set_x(rd, result);
    Ok(InstStatus::Continue)
}

/// CSEL / CSINC / CSINV / CSNEG (also covers CSET, CINC, CSETM, CINV via XZR encodings).
fn cond_select(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let sf   = bit(inst, 31);
    let op_  = bit(inst, 30); // 0=CSEL/CSINC, 1=CSINV/CSNEG
    let s    = bit(inst, 29);
    let rm   = bits(inst, 16, 5) as u8;
    let cond = bits(inst, 12, 4);
    let op2  = bits(inst, 10, 2);
    let rn   = bits(inst, 5, 5) as u8;
    let rd   = bits(inst, 0, 5) as u8;
    if s != 0 || op2 > 1 {
        return Err(Error::Decode { pc: em.current_pc, opcode: inst });
    }
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let ty   = if sf == 1 { Ty::U64 } else { Ty::U32 };

    let a = em.get_gpr(rn, size);
    let mut b = em.get_gpr(rm, size);

    // Compute the false-side value per variant:
    //   op_=0, op2=0 : CSEL  → b
    //   op_=0, op2=1 : CSINC → b + 1
    //   op_=1, op2=0 : CSINV → ~b
    //   op_=1, op2=1 : CSNEG → -b
    let one = em.const_u64(1);
    let all_ones = em.const_u64(if sf == 1 { !0u64 } else { 0xFFFF_FFFFu64 });
    b = match (op_, op2) {
        (0, 0) => b,
        (0, 1) => em.add(b, one, size),
        (1, 0) => em.eor(b, all_ones, size),
        (1, 1) => {
            let zero = em.const_u64(0);
            em.sub(zero, b, size)
        }
        _ => return Err(Error::Decode { pc: em.current_pc, opcode: inst }),
    };

    let nzcv = em.get_nzcv();
    let op = if sf == 1 { Op::Csel64 } else { Op::Csel32 };
    let result = em.push(Armlet::new(op, ty)
        .with_args(&[a, b, nzcv])
        .with_imm(cond as u64));

    em.set_x(rd, result);
    Ok(InstStatus::Continue)
}

/// MADD/MSUB/SMADDL/UMADDL/SMSUBL/UMSUBL/UMULH/SMULH and friends.
fn data_proc_3_source(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let sf  = bit(inst, 31);
    let op54 = bits(inst, 29, 2);
    let op31 = bits(inst, 21, 3);
    let rm   = bits(inst, 16, 5) as u8;
    let o0   = bit(inst, 15);
    let ra   = bits(inst, 10, 5) as u8;
    let rn   = bits(inst, 5, 5) as u8;
    let rd   = bits(inst, 0, 5) as u8;

    if op54 != 0 {
        return Err(Error::Decode { pc: em.current_pc, opcode: inst });
    }

    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let ty   = if sf == 1 { Ty::U64 } else { Ty::U32 };

    match (sf, op31, o0) {
        (_, 0b000, 0) => {
            // MADD: Rd = Ra + Rn*Rm
            let n = em.get_gpr(rn, size);
            let m = em.get_gpr(rm, size);
            let a = em.get_gpr(ra, size);
            let op = if sf == 1 { Op::Madd64 } else { Op::Madd32 };
            let res = em.push(Armlet::new(op, ty).with_args(&[n, m, a]));
            em.set_gpr(rd, res, size);
            Ok(InstStatus::Continue)
        }
        (_, 0b000, 1) => {
            // MSUB: Rd = Ra - Rn*Rm
            let n = em.get_gpr(rn, size);
            let m = em.get_gpr(rm, size);
            let a = em.get_gpr(ra, size);
            let op = if sf == 1 { Op::Msub64 } else { Op::Msub32 };
            let res = em.push(Armlet::new(op, ty).with_args(&[n, m, a]));
            em.set_gpr(rd, res, size);
            Ok(InstStatus::Continue)
        }
        (1, 0b010, 0) => {
            // SMULH
            let n = em.get_x(rn);
            let m = em.get_x(rm);
            let res = em.push(Armlet::new(Op::SMulH64, Ty::U64).with_args(&[n, m]));
            em.set_x(rd, res);
            Ok(InstStatus::Continue)
        }
        (1, 0b110, 0) => {
            // UMULH
            let n = em.get_x(rn);
            let m = em.get_x(rm);
            let res = em.push(Armlet::new(Op::UMulH64, Ty::U64).with_args(&[n, m]));
            em.set_x(rd, res);
            Ok(InstStatus::Continue)
        }
        _ => Err(Error::Unsupported { pc: em.current_pc, opcode: inst }),
    }
}

#[allow(dead_code)]
fn _cond_unused(c: Cond) { let _ = c; }
