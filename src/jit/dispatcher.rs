use crate::jit::context::CpuContext;

pub type JitFn = unsafe extern "C" fn(block_fn: u64, ctx: *mut CpuContext) -> u64;
