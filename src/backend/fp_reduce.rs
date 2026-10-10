//! Arm FPReduce for the four single-precision across-lane min/max forms.
//! Primary reference: Arm A64 ISA111181/2025-09_ASL1, FPReduce,
//! FPMin/FPMax and FPMinNum/FPMaxNum. The reduction tree is (0,1),(2,3),
//! then the two results, so NaN selection is deliberately stagewise.
//! These exact selections need no arithmetic rounding; baseline FPCR.DN/FZ
//! and cumulative IOC/IDC are implemented, not FEAT_AFP alternate handling.
use crate::jit::context::CpuContext;

const SIGN: u32 = 0x8000_0000;
const INFINITY: u32 = 0x7f80_0000;
const QUIET: u32 = 0x0040_0000;
const DEFAULT_NAN: u32 = INFINITY | QUIET;

fn unpack(value: u32, fpcr: u32, status: &mut u32) -> u32 {
    if fpcr & (1 << 24) != 0 && value & INFINITY == 0 && value & 0x007f_ffff != 0 {
        *status |= 1 << 7;
        value & SIGN
    } else {
        value
    }
}

pub(super) fn pair(
    n: u32,
    m: u32,
    minimum: bool,
    numeric: bool,
    fpcr: u32,
    status: &mut u32,
) -> u32 {
    let n = unpack(n, fpcr, status);
    let m = unpack(m, fpcr, status);
    let n_nan = n & !SIGN > INFINITY;
    let m_nan = m & !SIGN > INFINITY;
    let n_signaling = n_nan && n & QUIET == 0;
    let m_signaling = m_nan && m & QUIET == 0;
    if n_signaling || m_signaling {
        *status |= 1;
        let selected = if n_signaling { n } else { m };
        return if fpcr & (1 << 25) != 0 {
            DEFAULT_NAN
        } else {
            selected | QUIET
        };
    }
    if n_nan || m_nan {
        if numeric && n_nan != m_nan {
            return if n_nan { m } else { n };
        }
        let selected = if n_nan { n } else { m };
        return if fpcr & (1 << 25) != 0 {
            DEFAULT_NAN
        } else {
            selected
        };
    }
    if n & !SIGN == 0 && m & !SIGN == 0 {
        return if minimum {
            (n | m) & SIGN
        } else {
            (n & m) & SIGN
        };
    }
    let key = |value: u32| {
        if value & SIGN != 0 {
            !value
        } else {
            value ^ SIGN
        }
    };
    if (key(n) < key(m)) == minimum {
        n
    } else {
        m
    }
}

// mode0/1: FMAXV/FMINV, mode2/3: FMAXNMV/FMINNMV.
pub(super) unsafe extern "C" fn reduce(
    ctx: *mut CpuContext,
    low: u64,
    high: u64,
    mode: u64,
) -> u64 {
    let ctx = unsafe { &mut *ctx };
    let minimum = mode & 1 != 0;
    let numeric = mode & 2 != 0;
    let mut status = 0;
    let left = pair(
        low as u32,
        (low >> 32) as u32,
        minimum,
        numeric,
        ctx.fpcr,
        &mut status,
    );
    let right = pair(
        high as u32,
        (high >> 32) as u32,
        minimum,
        numeric,
        ctx.fpcr,
        &mut status,
    );
    let result = pair(left, right, minimum, numeric, ctx.fpcr, &mut status);
    ctx.fpsr |= status;
    result as u64
}
