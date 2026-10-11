//! Host functions behind `Op::CallHelper`.

use crate::jit::context::CpuContext;

pub type HelperFn = unsafe extern "C" fn(*mut CpuContext);

/// `interp::exec` on `helper_insn` / `helper_args`.
pub const INTERP: u32 = 0;

pub static HELPERS: [HelperFn; 1] = [interp_helper];

unsafe extern "C" fn interp_helper(ctx: *mut CpuContext) {
    let ctx = unsafe { &mut *ctx };
    let mut st = crate::interp::fp::St {
        fpcr: ctx.fpcr,
        fpsr: ctx.fpsr,
    };
    // Translation only emits encodings `interp::plan` accepted, which
    // `exec` always implements.
    let r = crate::interp::exec(ctx.helper_insn, &ctx.helper_args, &mut st).unwrap_or(0);
    ctx.fpsr = st.fpsr;
    ctx.helper_ret = [r as u64, (r >> 64) as u64];
}
