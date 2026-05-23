//! Dispatcher glue: the function pointer signature for emitted blocks and
//! the (future) inline-cached indirect-branch trampoline.

use crate::jit::context::CpuContext;

pub type JitFn = unsafe extern "C" fn(*mut CpuContext) -> u64;
