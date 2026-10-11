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
    /// Value observed by the last load-exclusive (zero-extended); a
    /// store-exclusive succeeds only if memory still holds it.
    pub exclusive_value: u64,
    /// High half of `exclusive_value` for 128-bit exclusive pairs.
    pub exclusive_value_hi: u64,
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
    /// PC of the first callback access skipped because an earlier access of
    /// the run faulted (0 when none was).  That access has not happened, so
    /// `JitConfig::resume_after_fault` resumes there.
    pub mem_fault_skip_pc: u64,
    /// Page-table fastmem: one byte per 4 KiB guest page.  Bit 0 set means a
    /// load may go straight to `mem_base + va`; bit 1 the same for stores.
    /// Any other access takes the `mem_read` / `mem_write` callbacks.  Only
    /// consulted when `JitConfig::fastmem_page_bits` is set.
    pub page_table: *const u8,
    /// Guest instructions generated code may still retire before returning;
    /// `run_bounded` sets it per slice and the default does not limit a
    /// directly invoked block.
    pub budget_remaining: i64,
    /// Direct-mapped dispatch table probed by generated indirect branches
    /// (`code_cache::DispatchSlot`).  Set by the `Jit` before each run.
    pub dispatch_table: *const u8,
    /// Atomic read-modify-write callback for accesses the inline fastmem
    /// path does not take (see `jit::atomic` for the protocol).  Must be
    /// atomic with respect to other host threads running guest code.
    pub mem_atomic: crate::jit::atomic::MemAtomicFn,
    /// Expected value for a `mem_atomic` compare-and-swap.
    pub atomic_expected: [u64; 2],
    /// Operands for an out-of-line instruction helper (`Op::CallHelper`).
    /// Generated code fills these right before the call; 32/64-bit values
    /// are zero-extended to 128 bits.
    pub helper_args: [[u64; 2]; 4],
    /// Result of the last instruction helper.
    pub helper_ret: [u64; 2],
    /// Encoding of the guest instruction the helper implements.
    pub helper_insn: u32,
}

/// `page_table` entry bit: loads may bypass the read callback.
pub const PAGE_DIRECT_READ: u8 = 1;
/// `page_table` entry bit: stores may bypass the write callback.
pub const PAGE_DIRECT_WRITE: u8 = 2;

unsafe impl Send for CpuContext {}

unsafe extern "C" fn default_read_cntpct(_ctx: *mut CpuContext) -> u64 {
    CpuContext::default_cntpct()
}

impl CpuContext {
    /// Read the same process-wide 19.2MHz monotonic clock used by the default
    /// CNTPCT_EL0/CNTVCT_EL0 callback, without borrowing or mutating a context.
    /// Embedders may use this to keep host services in the guest clock's epoch.
    /// This deliberately does not invoke a custom `read_cntpct` callback:
    /// custom callbacks retain their existing mutable-context contract.
    pub fn default_cntpct() -> u64 {
        use std::sync::OnceLock;
        use std::time::Instant;
        static START: OnceLock<Instant> = OnceLock::new();
        let elapsed = START.get_or_init(Instant::now).elapsed().as_nanos();
        let ticks = elapsed.saturating_mul(19_200_000) / 1_000_000_000;
        ticks.min(u128::from(u64::MAX)) as u64
    }
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
            exclusive_value: 0,
            exclusive_value_hi: 0,
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
            mem_fault_skip_pc: 0,
            page_table: core::ptr::null(),
            budget_remaining: i64::MAX,
            dispatch_table: core::ptr::null(),
            mem_atomic: crate::jit::atomic::default_mem_atomic,
            atomic_expected: [0; 2],
            helper_args: [[0; 2]; 4],
            helper_ret: [0; 2],
            helper_insn: 0,
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
    pub const fn exclusive_value() -> usize {
        offset_of!(CpuContext, exclusive_value)
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
    #[inline]
    pub const fn mem_fault_skip_pc() -> usize {
        offset_of!(CpuContext, mem_fault_skip_pc)
    }
    #[inline]
    pub const fn should_halt() -> usize {
        offset_of!(CpuContext, should_halt)
    }
    #[inline]
    pub const fn page_table() -> usize {
        offset_of!(CpuContext, page_table)
    }
    #[inline]
    pub const fn budget_remaining() -> usize {
        offset_of!(CpuContext, budget_remaining)
    }
    #[inline]
    pub const fn mem_atomic() -> usize {
        offset_of!(CpuContext, mem_atomic)
    }
    #[inline]
    pub const fn atomic_expected() -> usize {
        offset_of!(CpuContext, atomic_expected)
    }
    #[inline]
    pub const fn dispatch_table() -> usize {
        offset_of!(CpuContext, dispatch_table)
    }
    #[inline]
    pub const fn helper_arg(i: usize) -> usize {
        offset_of!(CpuContext, helper_args) + i * 16
    }
    #[inline]
    pub const fn helper_ret() -> usize {
        offset_of!(CpuContext, helper_ret)
    }
    #[inline]
    pub const fn helper_insn() -> usize {
        offset_of!(CpuContext, helper_insn)
    }
}

// Generated code tests `should_halt` and `mem_fault` with one 16-bit compare.
const _: () = assert!(
    core::mem::offset_of!(CpuContext, mem_fault) == core::mem::offset_of!(CpuContext, should_halt) + 1
);

const _: () = assert!(
    core::mem::offset_of!(CpuContext, exclusive_value_hi)
        == core::mem::offset_of!(CpuContext, exclusive_value) + 8
);
