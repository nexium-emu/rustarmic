use iced_x86::code_asm::*;

use crate::error::Result;

pub(super) const REVERSE_MATRIX: u64 = 0x8040_2010_0804_0201;
const IDENTITY_MATRIX: u64 = 0x0102_0408_1020_4080;

pub(super) fn left_shift_matrix(shift: u32) -> u64 {
    let mut matrix = 0;
    for output in 0..8u32 {
        if output >= shift {
            matrix |= (1u64 << (output - shift)) << (8 * (7 - output));
        }
    }
    matrix
}

pub(super) fn right_shift_matrix(shift: u32, signed: bool) -> u64 {
    let mut matrix = 0;
    for output in 0..8u32 {
        let input = output.saturating_add(shift);
        if signed || input < 8 {
            matrix |= (1u64 << input.min(7)) << (8 * (7 - output));
        }
    }
    matrix
}

fn shifted_diagonal(matrix: u64) -> Option<(i32, u8)> {
    let mut delta = None;
    let mut mask = 0u8;
    for output in 0..8u32 {
        let row = (matrix >> (8 * (7 - output))) as u8;
        if row == 0 {
            continue;
        }
        if !row.is_power_of_two() {
            return None;
        }
        let row_delta = row.trailing_zeros() as i32 - output as i32;
        if delta.is_some_and(|value| value != row_delta) {
            return None;
        }
        delta = Some(row_delta);
        mask |= 1 << output;
    }
    delta.map(|delta| (delta, mask))
}

fn signed_right_shift(matrix: u64) -> Option<i32> {
    (1..=8).find(|&shift| matrix == right_shift_matrix(shift, true)).map(|shift| shift as i32)
}

pub(crate) fn affine_cost(matrix: u64, addend: u8, q: bool, gfni: bool) -> u32 {
    let tail = if addend == 0 { 0 } else { 4 };
    let core = if gfni {
        4
    } else {
        match (matrix, signed_right_shift(matrix), shifted_diagonal(matrix)) {
            (IDENTITY_MATRIX, _, _) => tail,
            (0, _, _) => if addend == 0 { 1 } else { 3 },
            (REVERSE_MATRIX, _, _) => 27 + tail,
            (_, Some(_), _) => 7 + tail,
            (_, _, Some((delta, mask))) => {
                u32::from(delta != 0) + if mask == 0xff { 0 } else { 4 } + tail
            }
            _ => {
                let mut cost = 2 + tail;
                for output in 0..8u32 {
                    let row = (matrix >> (8 * (7 - output))) as u8;
                    cost += if row == 0 { 0 } else if row.is_power_of_two() { 8 } else { 19 };
                }
                cost
            }
        }
    };
    core + u32::from(!q)
}

fn broadcast(asm: &mut CodeAssembler, register: AsmRegisterXmm, value: u64) -> Result<()> {
    asm.mov(rax, value as i64)?;
    asm.movq(register, rax)?;
    asm.punpcklqdq(register, register)?;
    Ok(())
}

pub(super) fn emit(asm: &mut CodeAssembler, matrix: u64, addend: u8) -> Result<()> {
    if super::cpu_features::active_features().has_gfni {
        broadcast(asm, xmm1, matrix)?;
        asm.gf2p8affineqb(xmm0, xmm1, addend as u32)?;
        return Ok(());
    }

    match (matrix, signed_right_shift(matrix), shifted_diagonal(matrix)) {
        (IDENTITY_MATRIX, _, _) => {}
        (0, _, _) => {
            if addend == 0 {
                asm.pxor(xmm0, xmm0)?;
            } else {
                broadcast(asm, xmm0, u64::from_le_bytes([addend; 8]))?;
            }
            return Ok(());
        }
        (REVERSE_MATRIX, _, _) => {
            for (shift, mask) in [(1, 0x55), (2, 0x33), (4, 0x0f)] {
                asm.movdqa(xmm1, xmm0)?;
                asm.psrlw(xmm1, shift)?;
                broadcast(asm, xmm2, u64::from_le_bytes([mask; 8]))?;
                asm.pand(xmm1, xmm2)?;
                asm.pand(xmm0, xmm2)?;
                asm.psllw(xmm0, shift)?;
                asm.por(xmm0, xmm1)?;
            }
        }
        (_, Some(shift), _) => {
            asm.pmovsxbw(xmm1, xmm0)?;
            asm.psraw(xmm1, shift)?;
            asm.psrldq(xmm0, 8)?;
            asm.pmovsxbw(xmm2, xmm0)?;
            asm.psraw(xmm2, shift)?;
            asm.packsswb(xmm1, xmm2)?;
            asm.movdqa(xmm0, xmm1)?;
        }
        (_, _, Some((delta, mask))) => {
            if delta < 0 {
                asm.psllw(xmm0, -delta)?;
            } else if delta > 0 {
                asm.psrlw(xmm0, delta)?;
            }
            if mask != 0xff {
                broadcast(asm, xmm1, u64::from_le_bytes([mask; 8]))?;
                asm.pand(xmm0, xmm1)?;
            }
        }
        _ => {
            asm.pxor(xmm1, xmm1)?;
            for output in 0..8u32 {
                let row = (matrix >> (8 * (7 - output))) as u8;
                if row == 0 {
                    continue;
                }
                if row.is_power_of_two() {
                    asm.movdqa(xmm2, xmm0)?;
                    asm.psrlw(xmm2, row.trailing_zeros() as i32)?;
                } else {
                    broadcast(asm, xmm2, u64::from_le_bytes([row; 8]))?;
                    asm.pand(xmm2, xmm0)?;
                    for shift in [4, 2, 1] {
                        asm.movdqa(xmm3, xmm2)?;
                        asm.psrlw(xmm3, shift)?;
                        asm.pxor(xmm2, xmm3)?;
                    }
                }
                broadcast(asm, xmm3, 0x0101_0101_0101_0101)?;
                asm.pand(xmm2, xmm3)?;
                asm.psllw(xmm2, output as i32)?;
                asm.por(xmm1, xmm2)?;
            }
            asm.movdqa(xmm0, xmm1)?;
        }
    }
    if addend != 0 {
        broadcast(asm, xmm1, u64::from_le_bytes([addend; 8]))?;
        asm.pxor(xmm0, xmm1)?;
    }
    Ok(())
}
