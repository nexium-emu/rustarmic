//! Dispatcher glue: the function pointer signature for emitted blocks and
//! the (future) inline-cached indirect-branch trampoline.

use crate::jit::context::CpuContext;

/// The thunk-shaped entrypoint. Host code never calls a block directly any
/// more — it calls the shared thunk, passing the block's address as the
/// first argument and the CpuContext pointer as the second. See
/// [`crate::backend::thunk`] for the prologue/epilogue layout.
pub type JitFn = unsafe extern "C" fn(block_fn: u64, ctx: *mut CpuContext) -> u64;
