use crate::arch::{NUM_GPRS, NUM_VREGS};

#[repr(C, align(64))]
pub struct CpuContext {
    pub x: [u64; NUM_GPRS],
    pub sp: u64,
    pub v: [[u64; 2]; NUM_VREGS],
    pub pc: u64,
    pub mem_base: *mut u8,
    pub nzcv: u8,
    _pad0: [u8; 7],
}

unsafe impl Send for CpuContext {}

impl Default for CpuContext {
    fn default() -> Self {
        Self {
            x: [0; NUM_GPRS],
            sp: 0,
            v: [[0; 2]; NUM_VREGS],
            pc: 0,
            mem_base: core::ptr::null_mut(),
            nzcv: 0,
            _pad0: [0; 7],
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
