//! Guest CPU state.
//!
//! Layout is `#[repr(C)]` so the backend can address fields with constant
//! offsets relative to `CTX_REG`. Field offsets are exposed via the
//! `cpu_offsets` helpers — keep them in sync if you reorder fields.

use crate::arch::{NUM_GPRS, NUM_VREGS};

#[repr(C, align(64))]
pub struct CpuContext {
    /// X0..X30 (X31 is the zero register, not stored).
    pub x: [u64; NUM_GPRS],
    /// Stack pointer.
    pub sp: u64,
    /// Program counter.
    pub pc: u64,
    /// PSTATE.NZCV packed nibble.
    pub nzcv: u8,
    _pad0: [u8; 7],
    /// SIMD/FP V0..V31, each 128 bits.
    pub v: [[u64; 2]; NUM_VREGS],
    /// Optional base of a contiguous guest address space. If non-null, the
    /// backend can read/write memory as `[mem_base + guest_addr]` without
    /// trampolining out to a callback. Set to 0 to force callback usage.
    pub mem_base: *mut u8,
    /// Reserved for future use.
    pub _reserved: [u64; 4],
}

unsafe impl Send for CpuContext {}

impl Default for CpuContext {
    fn default() -> Self {
        Self {
            x: [0; NUM_GPRS],
            sp: 0,
            pc: 0,
            nzcv: 0,
            _pad0: [0; 7],
            v: [[0; 2]; NUM_VREGS],
            mem_base: core::ptr::null_mut(),
            _reserved: [0; 4],
        }
    }
}

pub mod cpu_offsets {
    use super::*;
    use core::mem::offset_of;

    #[inline] pub const fn xreg(i: usize) -> usize {
        offset_of!(CpuContext, x) + i * core::mem::size_of::<u64>()
    }
    #[inline] pub const fn sp()       -> usize { offset_of!(CpuContext, sp) }
    #[inline] pub const fn pc()       -> usize { offset_of!(CpuContext, pc) }
    #[inline] pub const fn nzcv()     -> usize { offset_of!(CpuContext, nzcv) }
    #[inline] pub const fn vreg(i: usize) -> usize {
        offset_of!(CpuContext, v) + i * 16
    }
    #[inline] pub const fn mem_base() -> usize { offset_of!(CpuContext, mem_base) }
}
