use crate::arch::{NUM_GPRS, NUM_VREGS};

#[repr(C, align(64))]
pub struct CpuContext {
    pub x: [u64; NUM_GPRS],
    pub sp: u64,
    pub v: [[u64; 2]; NUM_VREGS],
    pub pc: u64,
    pub exclusive_addr: u64,
    pub mem_base: *mut u8,
    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
    pub fpcr: u32,
    pub fpsr: u32,
    pub nzcv: u8,
    pub exclusive_size: u8,
    _pad0: [u8; 6],
}

unsafe impl Send for CpuContext {}

impl Default for CpuContext {
    fn default() -> Self {
        Self {
            x: [0; NUM_GPRS],
            sp: 0,
            v: [[0; 2]; NUM_VREGS],
            pc: 0,
            exclusive_addr: 0,
            mem_base: core::ptr::null_mut(),
            tpidr_el0: 0,
            tpidrro_el0: 0,
            fpcr: 0,
            fpsr: 0,
            nzcv: 0,
            exclusive_size: 0,
            _pad0: [0; 6],
        }
    }
}

pub mod cpu_offsets {
    use super::*;
    use core::mem::offset_of;

    #[inline] pub const fn xreg(i: usize) -> usize {
        offset_of!(CpuContext, x) + i * core::mem::size_of::<u64>()
    }
    #[inline] pub const fn sp()        -> usize { offset_of!(CpuContext, sp) }
    #[inline] pub const fn pc()        -> usize { offset_of!(CpuContext, pc) }
    #[inline] pub const fn nzcv()      -> usize { offset_of!(CpuContext, nzcv) }
    #[inline] pub const fn vreg(i: usize) -> usize {
        offset_of!(CpuContext, v) + i * 16
    }
    #[inline] pub const fn mem_base() -> usize { offset_of!(CpuContext, mem_base) }
    #[inline] pub const fn exclusive_addr() -> usize { offset_of!(CpuContext, exclusive_addr) }
    #[inline] pub const fn exclusive_size() -> usize { offset_of!(CpuContext, exclusive_size) }
    #[inline] pub const fn tpidr_el0() -> usize { offset_of!(CpuContext, tpidr_el0) }
    #[inline] pub const fn tpidrro_el0() -> usize { offset_of!(CpuContext, tpidrro_el0) }
    #[inline] pub const fn fpcr() -> usize { offset_of!(CpuContext, fpcr) }
    #[inline] pub const fn fpsr() -> usize { offset_of!(CpuContext, fpsr) }
}
