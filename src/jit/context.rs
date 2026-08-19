use crate::arch::{NUM_GPRS, NUM_VREGS};

pub type MemReadFn = unsafe extern "C" fn(*mut CpuContext, addr: u64, size: u8);
pub type MemWriteFn = unsafe extern "C" fn(*mut CpuContext, addr: u64, size: u8);

#[repr(C, align(64))]
pub struct CpuContext {
    pub x: [u64; NUM_GPRS],
    pub sp: u64,
    pub v: [[u64; 2]; NUM_VREGS],
    pub pc: u64,
    pub exclusive_addr: u64,
    pub mem_base: *mut u8,
    pub mem_base_va: u64,
    pub mem_size: u64,
    pub core_id: u64,
    pub stop_token: *const std::sync::atomic::AtomicBool,
    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
    pub cntfrq_el0: u64,
    pub read_cntpct: unsafe extern "C" fn(*mut CpuContext) -> u64,
    pub mem_read: MemReadFn,
    pub mem_write: MemWriteFn,
    pub io_value: [u64; 2],
    pub fpcr: u32,
    pub fpsr: u32,
    pub nzcv: u8,
    pub exclusive_size: u8,
    pub should_halt: u8,
    pub mem_fault: u8,
    pub mem_fault_access: u8,
    pub mem_fault_size: u8,
    pub mem_fault_cause: u8,
    pub mem_fault_addr: u64,
    pub mem_fault_pc: u64,
}

unsafe impl Send for CpuContext {}

unsafe extern "C" fn default_read_cntpct(_ctx: *mut CpuContext) -> u64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    let elapsed = START.get_or_init(Instant::now).elapsed().as_nanos();
    let ticks = elapsed.saturating_mul(19_200_000) / 1_000_000_000;
    ticks.min(u128::from(u64::MAX)) as u64
}

unsafe extern "C" fn default_mem_read(_: *mut CpuContext, addr: u64, size: u8) {
    panic!(
        "rustarmic: CpuContext.mem_read not installed (addr={:#x}, size={})",
        addr, size
    )
}
unsafe extern "C" fn default_mem_write(_: *mut CpuContext, addr: u64, size: u8) {
    panic!(
        "rustarmic: CpuContext.mem_write not installed (addr={:#x}, size={})",
        addr, size
    )
}

impl Default for CpuContext {
    fn default() -> Self {
        Self {
            x: [0; NUM_GPRS],
            sp: 0,
            v: [[0; 2]; NUM_VREGS],
            pc: 0,
            exclusive_addr: 0,
            mem_base: core::ptr::null_mut(),
            mem_base_va: 0,
            mem_size: 0,
            core_id: 0,
            stop_token: core::ptr::null(),
            tpidr_el0: 0,
            tpidrro_el0: 0,
            cntfrq_el0: 19_200_000,
            read_cntpct: default_read_cntpct,
            mem_read: default_mem_read,
            mem_write: default_mem_write,
            io_value: [0; 2],
            fpcr: 0,
            fpsr: 0,
            nzcv: 0,
            exclusive_size: 0,
            should_halt: 0,
            mem_fault: 0,
            mem_fault_access: 0,
            mem_fault_size: 0,
            mem_fault_cause: 0,
            mem_fault_addr: 0,
            mem_fault_pc: 0,
        }
    }
}

pub mod cpu_offsets {
    use super::*;
    use core::mem::offset_of;

    #[inline]
    pub const fn xreg(i: usize) -> usize {
        offset_of!(CpuContext, x) + i * core::mem::size_of::<u64>()
    }
    #[inline]
    pub const fn sp() -> usize {
        offset_of!(CpuContext, sp)
    }
    #[inline]
    pub const fn pc() -> usize {
        offset_of!(CpuContext, pc)
    }
    #[inline]
    pub const fn nzcv() -> usize {
        offset_of!(CpuContext, nzcv)
    }
    #[inline]
    pub const fn vreg(i: usize) -> usize {
        offset_of!(CpuContext, v) + i * 16
    }
    #[inline]
    pub const fn mem_base() -> usize {
        offset_of!(CpuContext, mem_base)
    }
    #[inline]
    pub const fn mem_base_va() -> usize {
        offset_of!(CpuContext, mem_base_va)
    }
    #[inline]
    pub const fn mem_size() -> usize {
        offset_of!(CpuContext, mem_size)
    }
    #[inline]
    pub const fn core_id() -> usize {
        offset_of!(CpuContext, core_id)
    }
    #[inline]
    pub const fn exclusive_addr() -> usize {
        offset_of!(CpuContext, exclusive_addr)
    }
    #[inline]
    pub const fn exclusive_size() -> usize {
        offset_of!(CpuContext, exclusive_size)
    }
    #[inline]
    pub const fn tpidr_el0() -> usize {
        offset_of!(CpuContext, tpidr_el0)
    }
    #[inline]
    pub const fn tpidrro_el0() -> usize {
        offset_of!(CpuContext, tpidrro_el0)
    }
    #[inline]
    pub const fn cntfrq_el0() -> usize {
        offset_of!(CpuContext, cntfrq_el0)
    }
    #[inline]
    pub const fn read_cntpct() -> usize {
        offset_of!(CpuContext, read_cntpct)
    }
    #[inline]
    pub const fn mem_read() -> usize {
        offset_of!(CpuContext, mem_read)
    }
    #[inline]
    pub const fn mem_write() -> usize {
        offset_of!(CpuContext, mem_write)
    }
    #[inline]
    pub const fn io_value() -> usize {
        offset_of!(CpuContext, io_value)
    }
    #[inline]
    pub const fn fpcr() -> usize {
        offset_of!(CpuContext, fpcr)
    }
    #[inline]
    pub const fn fpsr() -> usize {
        offset_of!(CpuContext, fpsr)
    }
    #[inline]
    pub const fn mem_fault() -> usize {
        offset_of!(CpuContext, mem_fault)
    }
    #[inline]
    pub const fn mem_fault_access() -> usize {
        offset_of!(CpuContext, mem_fault_access)
    }
    #[inline]
    pub const fn mem_fault_size() -> usize {
        offset_of!(CpuContext, mem_fault_size)
    }
    #[inline]
    pub const fn mem_fault_cause() -> usize {
        offset_of!(CpuContext, mem_fault_cause)
    }
    #[inline]
    pub const fn mem_fault_addr() -> usize {
        offset_of!(CpuContext, mem_fault_addr)
    }
    #[inline]
    pub const fn mem_fault_pc() -> usize {
        offset_of!(CpuContext, mem_fault_pc)
    }
}
