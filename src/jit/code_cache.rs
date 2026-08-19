#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

use region::{Allocation, Protection};

use crate::backend::{ChainSite, emit_thunk_bytes};
use crate::error::{Error, Result};

pub type ThunkFn =
    unsafe extern "C" fn(block_fn: u64, ctx: *mut crate::jit::context::CpuContext) -> u64;

pub struct CodeCache {
    region: Allocation,
    cursor: usize,
    capacity: usize,
    table: HashMap<u64, Entry>,
    pending: HashMap<u64, Vec<*mut u8>>,
    thunk: *const u8,
}

#[derive(Clone, Copy)]
struct Entry {
    host_ptr: *const u8,
    guest_start: u64,
    guest_end: u64,
    guest_insns: u32,
}

unsafe impl Send for CodeCache {}

impl CodeCache {
    pub fn new(bytes: usize) -> Result<Self> {
        let allocation = region::alloc(bytes, Protection::READ_WRITE)
            .map_err(|e| Error::HostAlloc(e.to_string()))?;
        let capacity = allocation.len();
        let mut this = Self {
            region: allocation,
            cursor: 0,
            capacity,
            table: HashMap::new(),
            pending: HashMap::new(),
            thunk: core::ptr::null(),
        };
        let thunk_bytes = emit_thunk_bytes()?;
        let thunk_ptr = this.append_raw(&thunk_bytes)?;
        this.thunk = thunk_ptr;
        Ok(this)
    }

    pub fn thunk(&self) -> *const u8 {
        self.thunk
    }

    pub fn lookup(&self, pc: u64) -> Option<*const u8> {
        self.table.get(&pc).map(|e| e.host_ptr)
    }

    pub fn lookup_meta(&self, pc: u64) -> Option<(*const u8, u32)> {
        self.table.get(&pc).map(|e| (e.host_ptr, e.guest_insns))
    }

    pub fn invalidate_range(&mut self, start: u64, len: u64) {
        let Some(end) = start.checked_add(len) else {
            self.table.clear();
            self.pending.clear();
            return;
        };
        self.table
            .retain(|_, entry| entry.guest_end <= start || entry.guest_start >= end);
        self.pending
            .retain(|&target_pc, _| target_pc < start || target_pc >= end);
    }

    /// Drop all translated blocks while retaining the executable allocation.
    /// The dispatcher retries compilation after a cache rollover instead of
    /// exposing CodeCacheFull to the guest.
    pub fn reset(&mut self) -> Result<()> {
        self.cursor = 0;
        self.table.clear();
        self.pending.clear();
        let thunk_bytes = emit_thunk_bytes()?;
        self.thunk = self.append_raw(&thunk_bytes)?;
        Ok(())
    }

    fn append_raw(&mut self, bytes: &[u8]) -> Result<*const u8> {
        let aligned_cursor = (self.cursor + 15) & !15;
        if aligned_cursor + bytes.len() > self.capacity {
            return Err(Error::CodeCacheFull);
        }
        let base = self.region.as_mut_ptr::<u8>();
        unsafe {
            // Keep the cache W^X: make the allocation writable only while a
            // new block is being published, then return it to RX before any
            // generated code can execute.
            region::protect(base, self.capacity, Protection::READ_WRITE)
                .map_err(|e| Error::HostAlloc(e.to_string()))?;
            let dst = base.add(aligned_cursor);
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
            self.cursor = aligned_cursor + bytes.len();
            region::protect(base, self.capacity, Protection::READ_EXECUTE)
                .map_err(|e| Error::HostAlloc(e.to_string()))?;
            #[cfg(target_arch = "x86_64")]
            {
                core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
            }
            Ok(dst as *const u8)
        }
    }

    pub fn install(
        &mut self,
        guest_pc: u64,
        guest_end: u64,
        guest_insns: u32,
        bytes: &[u8],
        chains: &[ChainSite],
        body_offset: u32,
        cache_block: bool,
    ) -> Result<*const u8> {
        let host_ptr = self.append_raw(bytes)?;
        let body_addr = unsafe { host_ptr.add(body_offset as usize) };
        if cache_block {
            self.table.insert(
                guest_pc,
                Entry {
                    host_ptr,
                    guest_start: guest_pc,
                    guest_end: guest_end.max(guest_pc.saturating_add(4)),
                    guest_insns: guest_insns.max(1),
                },
            );
        }

        // Direct patching is intentionally disabled until link slots and
        // execution epochs are in place.  Mutable inbound JMPs can otherwise
        // survive invalidation and jump into stale code.  The emitter emits a
        // normal fallback return at every chain site meanwhile.
        let _ = (body_addr, chains);

        Ok(host_ptr)
    }
}
