#![allow(clippy::too_many_arguments)]

use std::cell::Cell;
use std::collections::HashMap;

use region::{Allocation, Protection};

use crate::backend::{emit_thunk_bytes, ChainSite};
use crate::backend::thunk::{emit_thunk_bytes_for, ThunkKind};
use crate::error::{Error, Result};

pub type ThunkFn =
    unsafe extern "C" fn(block_fn: u64, ctx: *mut crate::jit::context::CpuContext) -> u64;

const DISPATCH_CACHE_ENTRIES: usize = 1024;

pub struct CodeCache {
    region: Allocation,
    cursor: usize,
    capacity: usize,
    table: HashMap<u64, Entry>,
    // Private to one Jit, just like the authoritative table. Exact PC tags
    // make collisions ordinary misses; only normal table entries are cached.
    dispatch_cache: [Cell<Option<(u64, DispatchMeta)>>; DISPATCH_CACHE_ENTRIES],
    pending: HashMap<u64, Vec<*mut u8>>,
    thunk: *const u8,
    xmm_free_thunk: *const u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DispatchMeta {
    pub host_ptr: *const u8,
    pub guest_insns: u32,
    pub thunk_kind: ThunkKind,
}

#[derive(Clone, Copy)]
struct Entry {
    host_ptr: *const u8,
    guest_start: u64,
    guest_end: u64,
    guest_insns: u32,
    thunk_kind: ThunkKind,
}

#[derive(Debug, PartialEq, Eq)]
struct PublicationRange {
    byte_start: usize,
    byte_end: usize,
    page_start: usize,
    page_len: usize,
}

fn publication_range(cursor: usize, len: usize, capacity: usize) -> Result<PublicationRange> {
    if len == 0 {
        return Err(Error::Backend("cannot publish an empty code block".into()));
    }
    let byte_start = cursor.checked_add(15).ok_or(Error::CodeCacheFull)? & !15;
    let byte_end = byte_start.checked_add(len).ok_or(Error::CodeCacheFull)?;
    if byte_end > capacity {
        return Err(Error::CodeCacheFull);
    }
    let page_size = region::page::size();
    let page_start = byte_start / page_size * page_size;
    let page_end = byte_end
        .checked_add(page_size - 1)
        .ok_or(Error::CodeCacheFull)?
        / page_size
        * page_size;
    if page_end > capacity {
        return Err(Error::CodeCacheFull);
    }
    Ok(PublicationRange {
        byte_start,
        byte_end,
        page_start,
        page_len: page_end - page_start,
    })
}

unsafe impl Send for CodeCache {}

#[cfg(all(test, target_os = "windows"))]
#[path = "xmm_thunk_tests.rs"]
mod xmm_thunk_tests;

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
            dispatch_cache: std::array::from_fn(|_| Cell::new(None)),
            pending: HashMap::new(),
            thunk: core::ptr::null(),
            xmm_free_thunk: core::ptr::null(),
        };
        let thunk_bytes = emit_thunk_bytes()?;
        let thunk_ptr = this.append_raw(&thunk_bytes)?;
        this.thunk = thunk_ptr;
        this.publish_xmm_free_thunk()?;
        Ok(this)
    }

    pub fn thunk(&self) -> *const u8 {
        self.thunk
    }

    pub(crate) fn thunk_for(&self, kind: ThunkKind) -> *const u8 {
        match kind {
            ThunkKind::Full => self.thunk,
            ThunkKind::XmmFree => self.xmm_free_thunk,
        }
    }

    fn publish_xmm_free_thunk(&mut self) -> Result<()> {
        // If publication fails, never leave an older epoch's lean pointer.
        // The freshly published full thunk is always a safe fallback.
        self.xmm_free_thunk = self.thunk;
        #[cfg(target_os = "windows")]
        {
            self.xmm_free_thunk = self.append_raw(&emit_thunk_bytes_for(ThunkKind::XmmFree)?)?;
        }
        #[cfg(not(target_os = "windows"))]
        {
            self.xmm_free_thunk = self.thunk;
        }
        Ok(())
    }

    pub fn lookup(&self, pc: u64) -> Option<*const u8> {
        self.lookup_meta(pc).map(|(host_ptr, _)| host_ptr)
    }

    #[inline(always)]
    pub fn lookup_meta(&self, pc: u64) -> Option<(*const u8, u32)> {
        self.lookup_dispatch_meta(pc).map(|entry| (entry.host_ptr, entry.guest_insns))
    }

    #[inline(always)]
    pub(crate) fn lookup_dispatch_meta(&self, pc: u64) -> Option<DispatchMeta> {
        // Fold higher PC bits into the index so aligned hot blocks do
        // not all occupy a small subset of slots. The full tag is still checked.
        let index = ((pc >> 2) ^ (pc >> 12)) as usize & (DISPATCH_CACHE_ENTRIES - 1);
        let slot = &self.dispatch_cache[index];
        if let Some((tag, metadata)) = slot.get() {
            if tag == pc {
                return Some(metadata);
            }
        }
        self.lookup_meta_miss(pc, slot)
    }

    #[inline(never)]
    fn lookup_meta_miss(
        &self,
        pc: u64,
        slot: &Cell<Option<(u64, DispatchMeta)>>,
    ) -> Option<DispatchMeta> {
        let entry = self.table.get(&pc)?;
        let metadata = DispatchMeta {
            host_ptr: entry.host_ptr,
            guest_insns: entry.guest_insns,
            thunk_kind: entry.thunk_kind,
        };
        slot.set(Some((pc, metadata)));
        Some(metadata)
    }

    fn clear_dispatch_cache(&self) {
        for slot in &self.dispatch_cache {
            slot.set(None);
        }
    }

    pub fn invalidate_range(&mut self, start: u64, len: u64) {
        // Clear even for overflow/full invalidation, before table pointers can
        // cease to be authoritative. Unaffected entries refill on demand.
        self.clear_dispatch_cache();
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
        self.clear_dispatch_cache();
        self.cursor = 0;
        self.table.clear();
        self.pending.clear();
        let thunk_bytes = emit_thunk_bytes()?;
        self.thunk = self.append_raw(&thunk_bytes)?;
        self.publish_xmm_free_thunk()?;
        Ok(())
    }

    fn append_raw(&mut self, bytes: &[u8]) -> Result<*const u8> {
        let range = publication_range(self.cursor, bytes.len(), self.capacity)?;
        let base = self.region.as_mut_ptr::<u8>();
        unsafe {
            let publish_pages = base.add(range.page_start);
            let dst = base.add(range.byte_start);
            // Keep publication W^X, but change only the pages containing the
            // appended bytes, not the entire (normally 256 MiB) allocation.
            // This cache is private to a Jit and no code from it executes while
            // that same Jit holds &mut self to install/reset a block. Published
            // pages outside this span remain RX; unused pages remain RW.
            region::protect(publish_pages, range.page_len, Protection::READ_WRITE)
                .map_err(|e| Error::HostAlloc(e.to_string()))?;
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
            region::protect(publish_pages, range.page_len, Protection::READ_EXECUTE)
                .map_err(|e| Error::HostAlloc(e.to_string()))?;
            self.cursor = range.byte_end;
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
        self.install_emitted(guest_pc, guest_end, guest_insns, bytes, chains, body_offset,
            cache_block, ThunkKind::Full)
    }

    pub(crate) fn install_emitted(
        &mut self,
        guest_pc: u64,
        guest_end: u64,
        guest_insns: u32,
        bytes: &[u8],
        chains: &[ChainSite],
        body_offset: u32,
        cache_block: bool,
        thunk_kind: ThunkKind,
    ) -> Result<*const u8> {
        let host_ptr = self.append_raw(bytes)?;
        let body_addr = unsafe { host_ptr.add(body_offset as usize) };
        if cache_block {
            // Includes replacing a translated block at an existing PC. A
            // transient budget-limited block never changes normal metadata.
            self.clear_dispatch_cache();
            self.table.insert(
                guest_pc,
                Entry {
                    host_ptr,
                    guest_start: guest_pc,
                    guest_end: guest_end.max(guest_pc.saturating_add(4)),
                    guest_insns: guest_insns.max(1),
                    thunk_kind,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn protection(cache: &CodeCache, offset: usize) -> Protection {
        let ptr = unsafe { cache.region.as_ptr::<u8>().add(offset) };
        region::query(ptr).unwrap().protection()
    }

    fn aligned(cursor: usize) -> usize {
        (cursor + 15) & !15
    }

    fn return_value(value: u32) -> [u8; 6] {
        let mut bytes = [0xb8, 0, 0, 0, 0, 0xc3]; // MOV EAX, imm32; RET.
        bytes[1..5].copy_from_slice(&value.to_le_bytes());
        bytes
    }

    unsafe fn call(cache: &CodeCache, block: *const u8) -> u64 {
        let thunk: ThunkFn = unsafe { core::mem::transmute(cache.thunk()) };
        // This test block is an independent host constant-return stub. It does
        // not access a CpuContext, guest memory or any external callbacks.
        unsafe { thunk(block as u64, core::ptr::null_mut()) }
    }

    #[test]
    fn checked_publication_ranges_align_and_cover_only_touched_pages() {
        let page = region::page::size();
        assert_eq!(
            publication_range(17, 6, page * 4).unwrap(),
            PublicationRange {
                byte_start: 32,
                byte_end: 38,
                page_start: 0,
                page_len: page,
            }
        );
        let crossing = publication_range(page - 16, 32, page * 4).unwrap();
        assert_eq!(crossing.page_start, 0);
        assert_eq!(crossing.page_len, page * 2);
        let boundary = publication_range(page, page, page * 2).unwrap();
        assert_eq!(boundary.page_start, page);
        assert_eq!(boundary.page_len, page);
        assert_eq!(boundary.byte_end, page * 2);
    }

    #[test]
    fn invalid_empty_overflow_and_capacity_ranges_are_rejected() {
        let page = region::page::size();
        assert!(matches!(
            publication_range(0, 0, page),
            Err(Error::Backend(_))
        ));
        for (cursor, len, capacity) in [
            (usize::MAX, 1, usize::MAX),
            (usize::MAX - 15, 32, usize::MAX),
            (0, usize::MAX, usize::MAX),
            (page - 16, 17, page),
            (page, 1, page),
        ] {
            assert!(matches!(
                publication_range(cursor, len, capacity),
                Err(Error::CodeCacheFull)
            ));
        }
    }

    #[test]
    fn publication_keeps_unused_and_unrelated_page_protections() {
        let page = region::page::size();
        let mut cache = CodeCache::new(page * 4).unwrap();
        assert_eq!(protection(&cache, 0), Protection::READ_EXECUTE);
        assert_eq!(protection(&cache, page), Protection::READ_WRITE);
        let sentinel = unsafe { cache.region.as_mut_ptr::<u8>().add(page * 3) };
        unsafe { region::protect(sentinel, page, Protection::READ).unwrap() };
        let first = cache.append_raw(&return_value(41)).unwrap();
        let second = cache.append_raw(&return_value(42)).unwrap();
        assert_eq!(unsafe { call(&cache, first) }, 41);
        assert_eq!(unsafe { call(&cache, second) }, 42);
        assert_eq!(protection(&cache, 0), Protection::READ_EXECUTE);
        assert_eq!(protection(&cache, page), Protection::READ_WRITE);
        assert_eq!(protection(&cache, page * 2), Protection::READ_WRITE);
        assert_eq!(protection(&cache, page * 3), Protection::READ);
    }

    #[test]
    fn exact_page_end_does_not_publish_the_following_page() {
        let page = region::page::size();
        let mut cache = CodeCache::new(page * 3).unwrap();
        let len = page - aligned(cache.cursor);
        let block = vec![0x90; len];
        let installed = cache.append_raw(&block).unwrap();
        assert_eq!(cache.cursor, page);
        assert_eq!(
            unsafe { core::slice::from_raw_parts(installed, len) },
            block
        );
        assert_eq!(protection(&cache, 0), Protection::READ_EXECUTE);
        assert_eq!(protection(&cache, page), Protection::READ_WRITE);
        let next = cache.append_raw(&return_value(43)).unwrap();
        assert_eq!(unsafe { call(&cache, next) }, 43);
        assert_eq!(protection(&cache, page), Protection::READ_EXECUTE);
        assert_eq!(protection(&cache, page * 2), Protection::READ_WRITE);
    }

    #[test]
    fn cross_page_publication_preserves_earlier_executable_blocks() {
        let page = region::page::size();
        let mut cache = CodeCache::new(page * 3).unwrap();
        let first = cache.append_raw(&return_value(44)).unwrap();
        let padding = page - 16 - aligned(cache.cursor);
        cache.append_raw(&vec![0x90; padding]).unwrap();
        let block = vec![0x75; 32];
        let crossing = cache.append_raw(&block).unwrap();
        assert_eq!(crossing as usize % page, page - 16);
        assert_eq!(unsafe { core::slice::from_raw_parts(crossing, 32) }, block);
        assert_eq!(protection(&cache, 0), Protection::READ_EXECUTE);
        assert_eq!(protection(&cache, page), Protection::READ_EXECUTE);
        assert_eq!(protection(&cache, page * 2), Protection::READ_WRITE);
        assert_eq!(unsafe { call(&cache, first) }, 44);
    }

    #[test]
    fn full_capacity_failure_leaves_existing_publication_unchanged() {
        let page = region::page::size();
        let mut cache = CodeCache::new(page).unwrap();
        let first = cache.append_raw(&return_value(45)).unwrap();
        let remaining = cache.capacity - aligned(cache.cursor);
        cache.append_raw(&vec![0x90; remaining]).unwrap();
        let cursor = cache.cursor;
        assert!(matches!(
            cache.append_raw(&[0xc3]),
            Err(Error::CodeCacheFull)
        ));
        assert_eq!(cache.cursor, cursor);
        assert_eq!(protection(&cache, 0), Protection::READ_EXECUTE);
        assert_eq!(unsafe { call(&cache, first) }, 45);
    }

    #[test]
    fn reset_invalidation_and_transient_publication_reexecute_safely() {
        let page = region::page::size();
        let mut cache = CodeCache::new(page * 3).unwrap();
        let first = cache
            .install(0x1000, 0x1004, 1, &return_value(46), &[], 0, true)
            .unwrap();
        let outside = cache
            .install(0x2000, 0x2004, 1, &return_value(47), &[], 0, true)
            .unwrap();
        cache.invalidate_range(0x1000, 4);
        assert!(cache.lookup(0x1000).is_none());
        assert_eq!(cache.lookup(0x2000), Some(outside));
        assert_eq!(unsafe { call(&cache, first) }, 46);
        let transient = cache
            .install(0x3000, 0x3004, 1, &return_value(48), &[], 0, false)
            .unwrap();
        assert!(cache.lookup(0x3000).is_none());
        assert_eq!(unsafe { call(&cache, transient) }, 48);
        cache.reset().unwrap();
        assert_eq!(cache.table.len(), 0);
        assert_eq!(cache.pending.len(), 0);
        let replacement = cache
            .install(0x1000, 0x1004, 1, &return_value(49), &[], 0, true)
            .unwrap();
        assert_eq!(unsafe { call(&cache, replacement) }, 49);
        assert_eq!(protection(&cache, 0), Protection::READ_EXECUTE);
        assert_eq!(protection(&cache, page), Protection::READ_WRITE);
    }
}
