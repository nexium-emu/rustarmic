//! Integer implementation of Arm SQXTN/UQXTN/SQXTUN's SatQ pseudocode.
//! Primary reference: Arm A64 ISA111181/2025-09_ASL1, SIMD-FP instructions
//! SQXTN/SQXTUN/UQXTN. Size3 is rejected by the frontend; saturation updates
//! only cumulative FPSR.QC and never clears existing status bits.
use crate::jit::context::CpuContext;

pub(super) unsafe extern "C" fn narrow(
    ctx: *mut CpuContext,
    low: u64,
    high: u64,
    mode: u64,
) -> u64 {
    let width = 8u32 << (mode & 3);
    let source_width = width * 2;
    let kind = (mode >> 2) & 3;
    let scalar = mode & 16 != 0;
    let source = low as u128 | ((high as u128) << 64);
    let source_mask = (1u128 << source_width) - 1;
    let destination_mask = (1u64 << width) - 1;
    let min = if kind == 0 {
        -(1i128 << (width - 1))
    } else {
        0
    };
    let max = if kind == 0 {
        (1i128 << (width - 1)) - 1
    } else {
        destination_mask as i128
    };
    let mut result = 0u64;
    let mut saturated = false;
    for lane in 0..if scalar { 1 } else { 64 / width } {
        let element = (source >> (lane * source_width)) & source_mask;
        let value = if kind == 1 {
            element as i128
        } else {
            ((element << (128 - source_width)) as i128) >> (128 - source_width)
        };
        let clamped = value.clamp(min, max);
        saturated |= clamped != value;
        result |= (clamped as u64 & destination_mask) << (lane * width);
    }
    if saturated {
        unsafe {
            (*ctx).fpsr |= 1 << 27;
        }
    }
    result
}
