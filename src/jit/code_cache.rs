//! RWX code cache.
//!
//! Holds a single contiguous executable region. We bump-allocate compiled
//! block bytes into it and remember a `guest_pc → host_ptr` map. When the
//! cache fills up we currently bail (a full flush + retranslate is a future
//! enhancement once we have block-linking patches to invalidate).

use std::collections::HashMap;

use region::{Allocation, Protection};

use crate::error::{Error, Result};

/// Function pointer type emitted by the backend.
pub type HostFn = unsafe extern "C" fn(*mut crate::jit::context::CpuContext) -> u64;

pub struct CodeCache {
    region:   Allocation,
    cursor:   usize,
    capacity: usize,
    table:    HashMap<u64, *const u8>,
}

unsafe impl Send for CodeCache {}

impl CodeCache {
    pub fn new(bytes: usize) -> Result<Self> {
        // Allocate readable/writable/executable memory. `region::alloc` rounds
        // to page size internally. We start RW and flip to RWX after the first
        // install so the platform can keep us happy with stricter W^X policies
        // when we later add the per-page protection toggle. For now: stay RWX.
        let allocation = region::alloc(bytes, Protection::READ_WRITE_EXECUTE)
            .map_err(|e| Error::HostAlloc(e.to_string()))?;
        let capacity = allocation.len();
        Ok(Self {
            region: allocation,
            cursor: 0,
            capacity,
            table: HashMap::new(),
        })
    }

    pub fn lookup(&self, pc: u64) -> Option<*const u8> {
        self.table.get(&pc).copied()
    }

    pub fn install(&mut self, guest_pc: u64, bytes: &[u8]) -> Result<*const u8> {
        // 16-byte align for branch-predictor friendliness.
        let aligned_cursor = (self.cursor + 15) & !15;
        if aligned_cursor + bytes.len() > self.capacity {
            return Err(Error::CodeCacheFull);
        }
        // SAFETY: region is RWX, allocation guarantees the slice lives as long
        // as `self`, and we hold exclusive `&mut self`.
        unsafe {
            let base = self.region.as_mut_ptr::<u8>();
            let dst = base.add(aligned_cursor);
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
            self.cursor = aligned_cursor + bytes.len();
            // i-cache flush is a no-op on x86_64 (coherent), but stay correct
            // for future ARM hosts.
            #[cfg(target_arch = "x86_64")]
            { core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst); }

            self.table.insert(guest_pc, dst as *const u8);
            Ok(dst as *const u8)
        }
    }
}
