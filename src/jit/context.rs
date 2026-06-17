use crate::arch::{NUM_GPRS, NUM_VREGS};

/// Memory-hook function pointer types. Single hook each for read/write, no
/// value argument — the value travels via `CpuContext.io_value` instead:
/// for reads the hook writes its result there, for writes the JIT stages
/// the value there before the call. This sidesteps the cross-platform XMM
/// argument-passing problem (vectorcall is nightly-only on stable Rust;
/// sysv64-on-Windows would trash callee-saved xmm6-15 and break the
/// regalloc's XMM spill pool). The cost is one extra mov per slow-path
/// access, which is dominated by the fastmem fast path in M1.5.
pub type MemReadFn  = unsafe extern "C" fn(*mut CpuContext, addr: u64, size: u8);
pub type MemWriteFn = unsafe extern "C" fn(*mut CpuContext, addr: u64, size: u8);

#[repr(C, align(64))]
pub struct CpuContext {
    pub x: [u64; NUM_GPRS],
    pub sp: u64,
    pub v: [[u64; 2]; NUM_VREGS],
    pub pc: u64,
    pub exclusive_addr: u64,
    /// Host pointer used by the soft-fastmem fast path. When `mem_size > 0`,
    /// emitted code that has fastmem enabled does `mov reg, [mem_base + (va -
    /// mem_base_va)]` for VAs in `[mem_base_va, mem_base_va + mem_size)`.
    pub mem_base: *mut u8,
    /// Guest VA that `mem_base` maps to. Together with `mem_size` defines
    /// the contiguous range eligible for direct host access.
    pub mem_base_va: u64,
    /// Length of the host region in bytes. `0` disables soft-fastmem even
    /// when `JitConfig::use_fastmem` is set; out-of-range accesses fall back
    /// to the fn-ptr handlers.
    pub mem_size: u64,
    pub tpidr_el0: u64,
    pub tpidrro_el0: u64,
    /// Guest-visible value of `CNTFRQ_EL0`. Embedder sets this to match its
    /// time source (Switch hardware: 19_200_000 Hz). The default below is the
    /// Switch value so libnx homebrew computes sensible nanoseconds.
    pub cntfrq_el0: u64,
    /// Called by emitted code on `MRS x, CNTPCT_EL0` / `MRS x, CNTVCT_EL0`.
    /// The default returns the host TSC (monotonic but rate-mismatched with
    /// `cntfrq_el0`); embedders that care about accurate scaling should
    /// install a callback that derives the value from their host clock.
    pub read_cntpct: unsafe extern "C" fn(*mut CpuContext) -> u64,
    /// Guest memory load hook. Called by emitted code for every guest load
    /// (scalar OR vector). `size` is the byte count (1, 2, 4, 8, or 16).
    /// The hook MUST write the loaded value into `self.io_value` (low `size`
    /// bytes; high bytes should be zero for sub-128-bit reads). Default panics
    /// — embedders MUST install a real handler before running data-bearing code.
    pub mem_read:  MemReadFn,
    /// Guest memory store hook. The JIT stages the value in `self.io_value`
    /// (low `size` bytes) before the call. Default panics.
    pub mem_write: MemWriteFn,
    /// 16-byte I/O buffer used to carry the value through every mem-hook
    /// call. `repr(C, align(16))` is inherited from CpuContext, so the
    /// emitter can use `movdqa` against `[CTX_REG + io_value_off]` for the
    /// 128-bit cases.
    pub io_value: [u64; 2],
    pub fpcr: u32,
    pub fpsr: u32,
    pub nzcv: u8,
    pub exclusive_size: u8,
    /// Cooperative halt flag. Set non-zero from a memory hook or another
    /// thread between blocks to make `Jit::run()` exit at the next block
    /// boundary with `ExitReason::Stopped`. Cleared by `Jit::run()` on
    /// entry. JIT-emitted code does not touch this field — it's polled by
    /// the run-loop in Rust.
    pub should_halt: u8,
    _pad0: [u8; 5],
}

unsafe impl Send for CpuContext {}

unsafe extern "C" fn default_read_cntpct(_ctx: *mut CpuContext) -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe { core::arch::x86_64::_rdtsc() }
    #[cfg(not(target_arch = "x86_64"))]
    { 0 }
}

unsafe extern "C" fn default_mem_read(_: *mut CpuContext, addr: u64, size: u8) {
    panic!("rustarmic: CpuContext.mem_read not installed (addr={:#x}, size={})", addr, size)
}
unsafe extern "C" fn default_mem_write(_: *mut CpuContext, addr: u64, size: u8) {
    panic!("rustarmic: CpuContext.mem_write not installed (addr={:#x}, size={})", addr, size)
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
            tpidr_el0: 0,
            tpidrro_el0: 0,
            cntfrq_el0: 19_200_000,
            read_cntpct: default_read_cntpct,
            mem_read:  default_mem_read,
            mem_write: default_mem_write,
            io_value:  [0; 2],
            fpcr: 0,
            fpsr: 0,
            nzcv: 0,
            exclusive_size: 0,
            should_halt: 0,
            _pad0: [0; 5],
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
    #[inline] pub const fn mem_base_va() -> usize { offset_of!(CpuContext, mem_base_va) }
    #[inline] pub const fn mem_size() -> usize { offset_of!(CpuContext, mem_size) }
    #[inline] pub const fn exclusive_addr() -> usize { offset_of!(CpuContext, exclusive_addr) }
    #[inline] pub const fn exclusive_size() -> usize { offset_of!(CpuContext, exclusive_size) }
    #[inline] pub const fn tpidr_el0() -> usize { offset_of!(CpuContext, tpidr_el0) }
    #[inline] pub const fn tpidrro_el0() -> usize { offset_of!(CpuContext, tpidrro_el0) }
    #[inline] pub const fn cntfrq_el0() -> usize { offset_of!(CpuContext, cntfrq_el0) }
    #[inline] pub const fn read_cntpct() -> usize { offset_of!(CpuContext, read_cntpct) }
    #[inline] pub const fn mem_read()  -> usize { offset_of!(CpuContext, mem_read) }
    #[inline] pub const fn mem_write() -> usize { offset_of!(CpuContext, mem_write) }
    #[inline] pub const fn io_value()  -> usize { offset_of!(CpuContext, io_value) }
    #[inline] pub const fn fpcr() -> usize { offset_of!(CpuContext, fpcr) }
    #[inline] pub const fn fpsr() -> usize { offset_of!(CpuContext, fpsr) }
}
