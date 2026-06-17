use std::collections::HashMap;

use region::{Allocation, Protection};

use crate::backend::{emit_thunk_bytes, ChainSite};
use crate::error::{Error, Result};

pub type ThunkFn = unsafe extern "C" fn(block_fn: u64, ctx: *mut crate::jit::context::CpuContext) -> u64;

pub struct CodeCache {
    region:   Allocation,
    cursor:   usize,
    capacity: usize,
    table:    HashMap<u64, Entry>,
    pending:  HashMap<u64, Vec<*mut u8>>,
    thunk:    *const u8,
}

#[derive(Clone, Copy)]
struct Entry {
    host_ptr:    *const u8,
    body_offset: u32,
}

unsafe impl Send for CodeCache {}

impl CodeCache {
    pub fn new(bytes: usize) -> Result<Self> {
        let allocation = region::alloc(bytes, Protection::READ_WRITE_EXECUTE)
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

    pub fn thunk(&self) -> *const u8 { self.thunk }

    pub fn lookup(&self, pc: u64) -> Option<*const u8> {
        self.table.get(&pc).map(|e| e.host_ptr)
    }

    pub fn invalidate_range(&mut self, start: u64, len: u64) {
        let end = start.wrapping_add(len);
        let in_range = |pc: u64| pc >= start && pc < end;
        self.table.retain(|&pc, _| !in_range(pc));
        self.pending.retain(|&target_pc, _| !in_range(target_pc));
    }

    fn append_raw(&mut self, bytes: &[u8]) -> Result<*const u8> {
        let aligned_cursor = (self.cursor + 15) & !15;
        if aligned_cursor + bytes.len() > self.capacity {
            return Err(Error::CodeCacheFull);
        }
        let ptr = unsafe {
            let base = self.region.as_mut_ptr::<u8>();
            let dst = base.add(aligned_cursor);
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
            self.cursor = aligned_cursor + bytes.len();
            #[cfg(target_arch = "x86_64")]
            { core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst); }
            dst as *const u8
        };
        Ok(ptr)
    }

    pub fn install(
        &mut self,
        guest_pc: u64,
        bytes: &[u8],
        chains: &[ChainSite],
        body_offset: u32,
    ) -> Result<*const u8> {
        let host_ptr = self.append_raw(bytes)?;
        let body_addr = unsafe { host_ptr.add(body_offset as usize) };
        self.table.insert(guest_pc, Entry { host_ptr, body_offset });

        if let Some(patches) = self.pending.remove(&guest_pc) {
            for patch_addr in patches {
                unsafe { patch_to_jmp(patch_addr, body_addr); }
            }
        }

        for c in chains {
            let patch_addr = unsafe { (host_ptr as *mut u8).add(c.patch_offset as usize) };
            if let Some(entry) = self.table.get(&c.target_pc) {
                let target_body = unsafe { entry.host_ptr.add(entry.body_offset as usize) };
                unsafe { patch_to_jmp(patch_addr, target_body); }
            } else {
                self.pending.entry(c.target_pc).or_default().push(patch_addr);
            }
        }

        Ok(host_ptr)
    }
}

unsafe fn patch_to_jmp(patch_addr: *mut u8, target: *const u8) {
    let rel = (target as isize).wrapping_sub((patch_addr as isize).wrapping_add(5));
    if rel < i32::MIN as isize || rel > i32::MAX as isize {
        return;
    }
    unsafe {
        patch_addr.write(0xE9);
        let rel_bytes = (rel as i32).to_le_bytes();
        for i in 0..4 {
            patch_addr.add(1 + i).write(rel_bytes[i]);
        }
    }
    #[cfg(target_arch = "x86_64")]
    { core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst); }
}
