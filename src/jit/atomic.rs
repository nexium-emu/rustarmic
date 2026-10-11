//! Guest atomic read-modify-write support (LSE atomics, CAS, store-exclusive).
//!
//! Generated code performs these with locked x86 instructions directly on
//! host memory when the fastmem guard admits the access.  Every other access
//! calls `CpuContext::mem_atomic`, which must perform the whole operation
//! atomically with respect to other host threads.  Embedders that can
//! translate the guest address to a host pointer should call [`host_atomic`];
//! the default callback ([`default_mem_atomic`]) serialises through
//! `mem_read` / `mem_write` under one process-wide lock, which is atomic with
//! respect to every other `mem_atomic` that uses it but not with respect to
//! plain stores.
//!
//! Callback protocol: `mem_atomic(ctx, addr, size, op)` with `size` in
//! {1, 2, 4, 8, 16}.  The operand (or, for [`AtomicOp::Cas`], the new value)
//! is in `ctx.io_value`; for `Cas` the expected value is in
//! `ctx.atomic_expected`.  On return `ctx.io_value` holds the old memory
//! value, zero-extended to 128 bits.  A `Cas` stored iff that old value
//! equals the expected one (compared over `size` bytes).  Faults are
//! reported like the other callbacks (`ctx.mem_fault`); nothing may be
//! written when the access faults.

use crate::jit::context::CpuContext;
use std::sync::Mutex;
use std::sync::atomic::{
    AtomicI8, AtomicI16, AtomicI32, AtomicI64, AtomicU8, AtomicU16, AtomicU32, AtomicU64,
    Ordering,
};

pub type MemAtomicFn = unsafe extern "C" fn(*mut CpuContext, addr: u64, size: u8, op: u8);

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicOp {
    Add = 0,
    /// `old & !operand` (LDCLR)
    Clr = 1,
    Eor = 2,
    /// `old | operand` (LDSET)
    Set = 3,
    Smax = 4,
    Smin = 5,
    Umax = 6,
    Umin = 7,
    Swp = 8,
    Cas = 9,
}

impl AtomicOp {
    pub fn from_u8(v: u8) -> Option<Self> {
        use AtomicOp::*;
        Some(match v {
            0 => Add,
            1 => Clr,
            2 => Eor,
            3 => Set,
            4 => Smax,
            5 => Smin,
            6 => Umax,
            7 => Umin,
            8 => Swp,
            9 => Cas,
            _ => return None,
        })
    }
}

fn mask(size: u8) -> u128 {
    if size >= 16 {
        u128::MAX
    } else {
        (1u128 << (u32::from(size) * 8)) - 1
    }
}

fn sext(v: u128, size: u8) -> i128 {
    let shift = 128 - u32::from(size) * 8;
    ((v << shift) as i128) >> shift
}

/// The value an RMW stores, given the old value; `None` for a failed `Cas`.
/// All values are zero-extended `size`-byte quantities.
pub fn apply(op: AtomicOp, size: u8, old: u128, operand: u128, expected: u128) -> Option<u128> {
    let m = mask(size);
    let (old, operand, expected) = (old & m, operand & m, expected & m);
    let new = match op {
        AtomicOp::Add => old.wrapping_add(operand),
        AtomicOp::Clr => old & !operand,
        AtomicOp::Eor => old ^ operand,
        AtomicOp::Set => old | operand,
        AtomicOp::Smax => {
            if sext(old, size) >= sext(operand, size) {
                old
            } else {
                operand
            }
        }
        AtomicOp::Smin => {
            if sext(old, size) <= sext(operand, size) {
                old
            } else {
                operand
            }
        }
        AtomicOp::Umax => old.max(operand),
        AtomicOp::Umin => old.min(operand),
        AtomicOp::Swp => operand,
        AtomicOp::Cas => {
            if old == expected {
                operand
            } else {
                return None;
            }
        }
    };
    Some(new & m)
}

/// Serialises RMWs that cannot be done lock-free (misaligned host pointers,
/// 128-bit without `cmpxchg16b`, and [`default_mem_atomic`]).
pub static FALLBACK_LOCK: Mutex<()> = Mutex::new(());

fn fallback_guard() -> std::sync::MutexGuard<'static, ()> {
    FALLBACK_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

macro_rules! rmw_int {
    ($ptr:expr, $op:expr, $operand:expr, $expected:expr, $u:ty, $au:ty, $i:ty, $ai:ty) => {{
        let a = unsafe { <$au>::from_ptr($ptr.cast::<$u>()) };
        let v = $operand as $u;
        let o = Ordering::SeqCst;
        let old: $u = match $op {
            AtomicOp::Add => a.fetch_add(v, o),
            AtomicOp::Clr => a.fetch_and(!v, o),
            AtomicOp::Eor => a.fetch_xor(v, o),
            AtomicOp::Set => a.fetch_or(v, o),
            AtomicOp::Umax => a.fetch_max(v, o),
            AtomicOp::Umin => a.fetch_min(v, o),
            AtomicOp::Smax => {
                let s = unsafe { <$ai>::from_ptr($ptr.cast::<$i>()) };
                s.fetch_max(v as $i, o) as $u
            }
            AtomicOp::Smin => {
                let s = unsafe { <$ai>::from_ptr($ptr.cast::<$i>()) };
                s.fetch_min(v as $i, o) as $u
            }
            AtomicOp::Swp => a.swap(v, o),
            AtomicOp::Cas => match a.compare_exchange($expected as $u, v, o, o) {
                Ok(x) | Err(x) => x,
            },
        };
        old as u128
    }};
}

/// Performs `op` atomically on `size` bytes at host address `ptr` and
/// returns the old value (zero-extended).  Naturally aligned 1/2/4/8-byte
/// accesses use host atomics, 16-byte ones `lock cmpxchg16b`; anything else
/// falls back to [`FALLBACK_LOCK`] (atomic only against other fallback
/// users).
///
/// # Safety
/// `ptr .. ptr + size` must be valid, writable host memory.
pub unsafe fn host_atomic(ptr: *mut u8, size: u8, op: AtomicOp, operand: u128, expected: u128) -> u128 {
    let aligned = (ptr as usize) & (usize::from(size).max(1) - 1) == 0;
    if aligned {
        match size {
            1 => return rmw_int!(ptr, op, operand, expected, u8, AtomicU8, i8, AtomicI8),
            2 => return rmw_int!(ptr, op, operand, expected, u16, AtomicU16, i16, AtomicI16),
            4 => return rmw_int!(ptr, op, operand, expected, u32, AtomicU32, i32, AtomicI32),
            8 => return rmw_int!(ptr, op, operand, expected, u64, AtomicU64, i64, AtomicI64),
            #[cfg(target_arch = "x86_64")]
            16 if std::arch::is_x86_feature_detected!("cmpxchg16b") => {
                return unsafe { rmw_128(ptr, op, operand, expected) };
            }
            _ => {}
        }
    }
    let _guard = fallback_guard();
    let n = usize::from(size).min(16);
    let mut buf = [0u8; 16];
    unsafe { core::ptr::copy_nonoverlapping(ptr, buf.as_mut_ptr(), n) };
    let old = u128::from_le_bytes(buf);
    if let Some(new) = apply(op, size, old, operand, expected) {
        let bytes = new.to_le_bytes();
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, n) };
    }
    old
}

/// One `lock cmpxchg16b`.  Returns (success, value observed).
///
/// # Safety
/// `ptr` must be 16-byte aligned, valid and writable; the CPU must support
/// `cmpxchg16b`.
#[cfg(target_arch = "x86_64")]
pub unsafe fn cmpxchg16b(ptr: *mut u8, expected: u128, new: u128) -> (bool, u128) {
    let (mut lo, mut hi) = (expected as u64, (expected >> 64) as u64);
    let ok: u8;
    unsafe {
        core::arch::asm!(
            "xchg {nlo}, rbx",
            "lock cmpxchg16b xmmword ptr [{p}]",
            "sete {ok}",
            "mov rbx, {nlo}",
            p = in(reg) ptr,
            nlo = inout(reg) new as u64 => _,
            ok = out(reg_byte) ok,
            inout("rax") lo,
            inout("rdx") hi,
            in("rcx") (new >> 64) as u64,
            options(nostack),
        );
    }
    (ok != 0, (u128::from(hi) << 64) | u128::from(lo))
}

#[cfg(target_arch = "x86_64")]
unsafe fn rmw_128(ptr: *mut u8, op: AtomicOp, operand: u128, expected: u128) -> u128 {
    // Seed with a plain read; a torn value only costs one retry.
    let mut cur = unsafe { (ptr as *const u128).read_volatile() };
    loop {
        let Some(new) = apply(op, 16, cur, operand, expected) else {
            // Failed CAS: confirm the value atomically (cmpxchg with
            // itself never changes memory).
            let (ok, seen) = unsafe { cmpxchg16b(ptr, cur, cur) };
            if ok {
                return cur;
            }
            cur = seen;
            continue;
        };
        let (ok, seen) = unsafe { cmpxchg16b(ptr, cur, new) };
        if ok {
            return cur;
        }
        cur = seen;
    }
}

/// Default `mem_atomic`: read-modify-write through `mem_read` / `mem_write`
/// under [`FALLBACK_LOCK`].
///
/// # Safety
/// Called by generated code with a valid context.
pub unsafe extern "C" fn default_mem_atomic(ctx: *mut CpuContext, addr: u64, size: u8, op: u8) {
    let _guard = fallback_guard();
    unsafe {
        let operand = (*ctx).io_value;
        let expected = (*ctx).atomic_expected;
        ((*ctx).mem_read)(ctx, addr, size);
        let old = (*ctx).io_value;
        if (*ctx).mem_fault == 0 {
            let to128 = |v: [u64; 2]| u128::from(v[0]) | (u128::from(v[1]) << 64);
            let op = AtomicOp::from_u8(op).unwrap_or(AtomicOp::Cas);
            if let Some(new) = apply(op, size, to128(old), to128(operand), to128(expected)) {
                (*ctx).io_value = [new as u64, (new >> 64) as u64];
                ((*ctx).mem_write)(ctx, addr, size);
            }
        }
        let m = mask(size);
        let old = (u128::from(old[0]) | (u128::from(old[1]) << 64)) & m;
        (*ctx).io_value = [old as u64, (old >> 64) as u64];
    }
}

/// `mem_atomic` helper for embedders: performs the operation described by
/// the context (see the module docs) on host memory at `ptr`.
///
/// # Safety
/// `ctx` valid; `ptr .. ptr + size` valid writable host memory.
pub unsafe fn context_host_atomic(ctx: *mut CpuContext, ptr: *mut u8, size: u8, op: u8) {
    unsafe {
        let to128 = |v: [u64; 2]| u128::from(v[0]) | (u128::from(v[1]) << 64);
        let operand = to128((*ctx).io_value);
        let expected = to128((*ctx).atomic_expected);
        let op = AtomicOp::from_u8(op).unwrap_or(AtomicOp::Cas);
        let old = host_atomic(ptr, size, op, operand, expected) & mask(size);
        (*ctx).io_value = [old as u64, (old >> 64) as u64];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_respects_width_and_signedness() {
        assert_eq!(apply(AtomicOp::Add, 1, 0xFF, 1, 0), Some(0));
        assert_eq!(apply(AtomicOp::Smax, 1, 0x80, 0x01, 0), Some(0x01));
        assert_eq!(apply(AtomicOp::Umax, 1, 0x80, 0x01, 0), Some(0x80));
        assert_eq!(apply(AtomicOp::Smin, 2, 0x8000, 0x7FFF, 0), Some(0x8000));
        assert_eq!(apply(AtomicOp::Cas, 4, 5, 9, 5), Some(9));
        assert_eq!(apply(AtomicOp::Cas, 4, 5, 9, 6), None);
        assert_eq!(apply(AtomicOp::Cas, 1, 0x105, 9, 0x205), Some(9));
    }

    #[test]
    fn host_atomic_all_widths() {
        #[repr(align(16))]
        struct A([u8; 32]);
        let mut m = A([0; 32]);
        let p = m.0.as_mut_ptr();
        unsafe {
            for size in [1u8, 2, 4, 8, 16] {
                assert_eq!(host_atomic(p, size, AtomicOp::Swp, 0x11, 0), 0);
                assert_eq!(host_atomic(p, size, AtomicOp::Add, 1, 0), 0x11);
                assert_eq!(host_atomic(p, size, AtomicOp::Cas, 7, 0x12), 0x12);
                assert_eq!(host_atomic(p, size, AtomicOp::Cas, 9, 0x12), 7);
                assert_eq!(host_atomic(p, size, AtomicOp::Swp, 0, 0), 7);
                // Misaligned: lock fallback.
                assert_eq!(host_atomic(p.add(1), size, AtomicOp::Add, 3, 0), 0);
                assert_eq!(host_atomic(p.add(1), size, AtomicOp::Swp, 0, 0), 3);
            }
            let big = (5u128 << 64) | 3;
            assert_eq!(host_atomic(p, 16, AtomicOp::Swp, big, 0), 0);
            assert_eq!(host_atomic(p, 16, AtomicOp::Cas, 1, big), big);
            assert_eq!(host_atomic(p, 16, AtomicOp::Swp, 0, 0), 1);
        }
    }
}
