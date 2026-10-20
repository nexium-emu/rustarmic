#![allow(clippy::too_many_arguments)]

use std::cell::Cell;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use region::{Allocation, Protection};
use smallvec::SmallVec;

use crate::backend::thunk::{emit_thunk_bytes_for, ThunkKind};
use crate::backend::{emit_thunk_bytes, ChainSite};
use crate::error::{Error, Result};

pub type ThunkFn =
    unsafe extern "C" fn(block_fn: u64, ctx: *mut crate::jit::context::CpuContext) -> u64;

/// Folded-multiply hasher for guest-PC keys; SipHash costs more than the rest
/// of a dispatcher lookup and a collision only costs a slower probe.
#[derive(Default, Clone, Copy)]
pub struct PcHasher(u64);

impl Hasher for PcHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }
    #[inline]
    fn write_u64(&mut self, v: u64) {
        let product = u128::from(self.0 ^ v) * 0x9E37_79B9_7F4A_7C15;
        self.0 = (product as u64) ^ ((product >> 64) as u64);
    }
    #[inline]
    fn write_u32(&mut self, v: u32) {
        self.write_u64(u64::from(v));
    }
}

type PcMap<K, V> = HashMap<K, V, BuildHasherDefault<PcHasher>>;

/// One slot of the direct-mapped dispatch table probed by the Rust
/// dispatcher and by generated indirect branches before the hash map.
/// `pc == EMPTY_PC` marks an empty slot.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DispatchSlot {
    pub pc: u64,
    pub host: u64,
}

pub const DISPATCH_BITS: u32 = 16;
pub const DISPATCH_SLOTS: usize = 1 << DISPATCH_BITS;
pub const DISPATCH_MASK: u32 = (DISPATCH_SLOTS as u32) - 1;
/// Never a valid 4-byte aligned PC; an indirect branch to it hits the miss
/// stub, which returns to the dispatcher.
pub const EMPTY_PC: u64 = u64::MAX;

/// Slot of `pc`: higher PC bits are folded in so blocks a multiple of the
/// table span apart do not all share one slot.
#[inline]
pub fn dispatch_index(pc: u64) -> usize {
    (((pc >> 2) ^ (pc >> (2 + DISPATCH_BITS))) as u32 & DISPATCH_MASK) as usize
}

/// Byte offset of the rel32 inside a chain site (`E9 rel32`).
const SITE_REL_OFFSET: usize = 1;
const SITE_LEN: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DispatchMeta {
    pub host_ptr: *const u8,
    pub guest_insns: u32,
    pub thunk_kind: ThunkKind,
}

pub struct CodeCache {
    memory: CodeMemory,
    cursor: usize,
    capacity: usize,
    page: usize,
    table: PcMap<u64, Entry>,
    /// Budget-capped blocks keyed by (pc, instruction cap); never linked to
    /// and never in the dispatch table.
    capped: PcMap<(u64, u32), Capped>,
    /// Every registered chain site that jumps to a guest PC, linked or not.
    /// A site is linked exactly while its target is in `table`.
    links: PcMap<u64, SmallVec<[usize; 4]>>,
    dispatch: Box<[Cell<DispatchSlot>]>,
    /// Instruction count and thunk of the block in the same `dispatch` slot.
    dispatch_meta: Box<[Cell<(u32, ThunkKind)>]>,
    thunk: *const u8,
    xmm_free_thunk: *const u8,
    miss_stub: *const u8,
    patches: Vec<(usize, i32)>,
}

struct Entry {
    host_ptr: *const u8,
    ranges: SmallVec<[(u64, u64); 4]>,
    /// (target pc, absolute site address) for each chain site in the block.
    out_sites: SmallVec<[(u64, usize); 2]>,
    guest_insns: u32,
    thunk_kind: ThunkKind,
}

struct Capped {
    host_ptr: *const u8,
    ranges: SmallVec<[(u64, u64); 4]>,
}

fn overlaps(ranges: &[(u64, u64)], start: u64, end: u64) -> bool {
    ranges.iter().any(|&(s, e)| s < end && e > start)
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

enum CodeMemory {
    Allocated { allocation: Allocation, rwx: bool },
    #[cfg(windows)]
    Reserved(ReservedCode),
}

impl CodeMemory {
    fn new(bytes: usize, rwx: bool) -> Result<Self> {
        #[cfg(windows)]
        {
            if rwx {
                return ReservedCode::reserve(bytes).map(Self::Reserved);
            }
        }
        let protection = if rwx {
            Protection::READ_WRITE_EXECUTE
        } else {
            Protection::READ_WRITE
        };
        let allocation =
            region::alloc(bytes, protection).map_err(|e| Error::HostAlloc(e.to_string()))?;
        Ok(Self::Allocated { allocation, rwx })
    }

    fn base(&self) -> *mut u8 {
        match self {
            Self::Allocated { allocation, .. } => allocation.as_ptr::<u8>().cast_mut(),
            #[cfg(windows)]
            Self::Reserved(code) => code.ptr,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Allocated { allocation, .. } => allocation.len(),
            #[cfg(windows)]
            Self::Reserved(code) => code.len,
        }
    }

    fn rwx(&self) -> bool {
        match self {
            Self::Allocated { rwx, .. } => *rwx,
            #[cfg(windows)]
            Self::Reserved(_) => true,
        }
    }

    fn prepare(&mut self, end: usize) -> Result<()> {
        match self {
            Self::Allocated { .. } => Ok(()),
            #[cfg(windows)]
            Self::Reserved(code) => code.commit_to(end),
        }
    }
}

#[cfg(windows)]
struct ReservedCode {
    ptr: *mut u8,
    len: usize,
    committed: usize,
}

#[cfg(windows)]
mod reserved_os {
    use core::ffi::c_void;

    pub const COMMIT_CHUNK: usize = 2 << 20;
    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_EXECUTE_READWRITE: u32 = 0x40;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
        fn VirtualFree(addr: *mut c_void, size: usize, kind: u32) -> i32;
    }

    pub fn reserve(len: usize) -> Option<*mut u8> {
        let p = unsafe { VirtualAlloc(core::ptr::null_mut(), len, MEM_RESERVE, PAGE_NOACCESS) };
        (!p.is_null()).then_some(p.cast())
    }

    pub fn commit(ptr: *mut u8, len: usize) -> bool {
        !unsafe { VirtualAlloc(ptr.cast(), len, MEM_COMMIT, PAGE_EXECUTE_READWRITE) }.is_null()
    }

    pub fn release(ptr: *mut u8) {
        unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) };
    }
}

#[cfg(windows)]
impl ReservedCode {
    fn reserve(bytes: usize) -> Result<Self> {
        let page = region::page::size();
        let len = bytes.checked_add(page - 1).ok_or(Error::CodeCacheFull)? / page * page;
        let ptr = reserved_os::reserve(len)
            .ok_or_else(|| Error::HostAlloc("code cache reservation failed".into()))?;
        Ok(Self {
            ptr,
            len,
            committed: 0,
        })
    }

    fn commit_to(&mut self, end: usize) -> Result<()> {
        if end <= self.committed {
            return Ok(());
        }
        if end > self.len {
            return Err(Error::CodeCacheFull);
        }
        let target = end
            .div_ceil(reserved_os::COMMIT_CHUNK)
            .saturating_mul(reserved_os::COMMIT_CHUNK)
            .min(self.len);
        let start = unsafe { self.ptr.add(self.committed) };
        if !reserved_os::commit(start, target - self.committed) {
            return Err(Error::HostAlloc("code cache commit failed".into()));
        }
        self.committed = target;
        Ok(())
    }
}

#[cfg(windows)]
impl Drop for ReservedCode {
    fn drop(&mut self) {
        reserved_os::release(self.ptr);
    }
}

unsafe impl Send for CodeCache {}

#[cfg(all(test, target_os = "windows"))]
#[path = "xmm_thunk_tests.rs"]
mod xmm_thunk_tests;

impl CodeCache {
    pub fn new(bytes: usize) -> Result<Self> {
        Self::with_rwx(bytes, false)
    }

    pub fn with_rwx(bytes: usize, rwx: bool) -> Result<Self> {
        let memory = CodeMemory::new(bytes, rwx)?;
        let capacity = memory.len();
        if capacity > i32::MAX as usize {
            return Err(Error::HostAlloc(
                "code cache must be smaller than 2 GiB (rel32 block links)".into(),
            ));
        }
        let empty = DispatchSlot {
            pc: EMPTY_PC,
            host: 0,
        };
        let mut this = Self {
            memory,
            cursor: 0,
            capacity,
            page: region::page::size(),
            table: PcMap::default(),
            capped: PcMap::default(),
            links: PcMap::default(),
            dispatch: (0..DISPATCH_SLOTS).map(|_| Cell::new(empty)).collect(),
            dispatch_meta: (0..DISPATCH_SLOTS)
                .map(|_| Cell::new((0, ThunkKind::Full)))
                .collect(),
            thunk: core::ptr::null(),
            xmm_free_thunk: core::ptr::null(),
            miss_stub: core::ptr::null(),
            patches: Vec::new(),
        };
        this.install_runtime_stubs()?;
        Ok(this)
    }

    fn install_runtime_stubs(&mut self) -> Result<()> {
        let thunk_bytes = emit_thunk_bytes()?;
        self.thunk = self.append_raw(&thunk_bytes)?;
        self.publish_xmm_free_thunk()?;
        // `ret` with RAX still holding the branch target: a dispatch-table
        // miss returns to the dispatcher.
        self.miss_stub = self.append_raw(&[0xC3])?;
        self.clear_dispatch();
        Ok(())
    }

    fn clear_dispatch(&self) {
        let empty = DispatchSlot {
            pc: EMPTY_PC,
            host: self.miss_stub as u64,
        };
        for slot in self.dispatch.iter() {
            slot.set(empty);
        }
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

    /// Address of the direct-mapped dispatch table for generated code.
    pub fn dispatch_table(&self) -> *const DispatchSlot {
        self.dispatch.as_ptr().cast()
    }

    #[inline]
    pub fn lookup(&self, pc: u64) -> Option<*const u8> {
        self.lookup_dispatch_meta(pc).map(|entry| entry.host_ptr)
    }

    #[inline(always)]
    pub fn lookup_meta(&self, pc: u64) -> Option<(*const u8, u32)> {
        self.lookup_dispatch_meta(pc)
            .map(|entry| (entry.host_ptr, entry.guest_insns))
    }

    #[inline(always)]
    pub(crate) fn lookup_dispatch_meta(&self, pc: u64) -> Option<DispatchMeta> {
        let index = dispatch_index(pc);
        let slot = self.dispatch[index].get();
        if slot.pc == pc {
            let (guest_insns, thunk_kind) = self.dispatch_meta[index].get();
            return Some(DispatchMeta {
                host_ptr: slot.host as *const u8,
                guest_insns,
                thunk_kind,
            });
        }
        self.lookup_meta_miss(pc, index)
    }

    #[inline(never)]
    fn lookup_meta_miss(&self, pc: u64, index: usize) -> Option<DispatchMeta> {
        let entry = self.table.get(&pc)?;
        self.dispatch[index].set(DispatchSlot {
            pc,
            host: entry.host_ptr as u64,
        });
        self.dispatch_meta[index].set((entry.guest_insns, entry.thunk_kind));
        Some(DispatchMeta {
            host_ptr: entry.host_ptr,
            guest_insns: entry.guest_insns,
            thunk_kind: entry.thunk_kind,
        })
    }

    pub fn contains(&self, pc: u64) -> bool {
        self.table.contains_key(&pc)
    }

    pub fn lookup_capped(&self, pc: u64, cap: u32) -> Option<*const u8> {
        self.capped.get(&(pc, cap)).map(|c| c.host_ptr)
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Bytes of the cache in use.
    pub fn used_bytes(&self) -> usize {
        self.cursor
    }

    /// Number of chain sites currently patched to jump to `pc`'s block.
    pub fn inbound_links(&self, pc: u64) -> usize {
        if !self.table.contains_key(&pc) {
            return 0;
        }
        self.links.get(&pc).map_or(0, |s| s.len())
    }

    pub fn invalidate_range(&mut self, start: u64, len: u64) {
        let (start, end) = match start.checked_add(len) {
            Some(end) => (start, end),
            None => (0, u64::MAX),
        };
        let all = start == 0 && end == u64::MAX;
        let doomed: Vec<u64> = self
            .table
            .iter()
            .filter(|(_, e)| all || overlaps(&e.ranges, start, end))
            .map(|(&pc, _)| pc)
            .collect();
        for pc in doomed {
            self.remove_block(pc);
        }
        self.capped
            .retain(|_, c| !all && !overlaps(&c.ranges, start, end));
        self.flush_patches();
    }

    /// Unpublishes one block: clears its dispatch slot, unlinks every site
    /// that jumps to it and forgets its own outgoing sites.  The code bytes
    /// stay in place until the next reset.
    fn remove_block(&mut self, pc: u64) {
        let Some(entry) = self.table.remove(&pc) else {
            return;
        };
        let slot = &self.dispatch[dispatch_index(pc)];
        if slot.get().pc == pc {
            slot.set(DispatchSlot {
                pc: EMPTY_PC,
                host: self.miss_stub as u64,
            });
        }
        if let Some(sites) = self.links.get(&pc) {
            for &site in sites {
                self.patches.push((site, 0));
            }
        }
        for (target, site) in entry.out_sites {
            if let Some(sites) = self.links.get_mut(&target) {
                sites.retain(|s| *s != site);
                if sites.is_empty() {
                    self.links.remove(&target);
                }
            }
        }
    }

    /// Drop all translated blocks while retaining the executable allocation.
    /// The dispatcher retries compilation after a cache rollover instead of
    /// exposing CodeCacheFull to the guest.  Must only run while no
    /// generated code is on the stack.
    pub fn reset(&mut self) -> Result<()> {
        self.cursor = 0;
        self.table.clear();
        self.capped.clear();
        self.links.clear();
        self.patches.clear();
        self.install_runtime_stubs()
    }

    /// Makes the pages of `[addr, addr + len)` writable, runs `f`, and
    /// returns them to read+execute.
    fn with_writable<R>(&mut self, addr: usize, len: usize, f: impl FnOnce() -> R) -> Result<R> {
        if self.memory.rwx() {
            return Ok(f());
        }
        let lo = addr & !(self.page - 1);
        let hi = (addr + len + self.page - 1) & !(self.page - 1);
        unsafe {
            region::protect(lo as *const u8, hi - lo, Protection::READ_WRITE)
                .map_err(|e| Error::HostAlloc(e.to_string()))?;
        }
        let out = f();
        unsafe {
            region::protect(lo as *const u8, hi - lo, Protection::READ_EXECUTE)
                .map_err(|e| Error::HostAlloc(e.to_string()))?;
        }
        Ok(out)
    }

    fn append_raw(&mut self, bytes: &[u8]) -> Result<*const u8> {
        let range = publication_range(self.cursor, bytes.len(), self.capacity)?;
        self.memory.prepare(range.page_start + range.page_len)?;
        let base = self.memory.base();
        let rwx = self.memory.rwx();
        unsafe {
            let publish_pages = base.add(range.page_start);
            let dst = base.add(range.byte_start);
            if !rwx {
                region::protect(publish_pages, range.page_len, Protection::READ_WRITE)
                    .map_err(|e| Error::HostAlloc(e.to_string()))?;
            }
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
            if !rwx {
                region::protect(publish_pages, range.page_len, Protection::READ_EXECUTE)
                    .map_err(|e| Error::HostAlloc(e.to_string()))?;
            }
            self.cursor = range.byte_end;
            core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
            Ok(dst as *const u8)
        }
    }

    fn queue_link(&mut self, site: usize, target: *const u8) {
        let rel = (target as isize) - (site as isize + SITE_LEN as isize);
        let rel = i32::try_from(rel).expect("code cache larger than rel32 range");
        self.patches.push((site, rel));
    }

    /// Writes every queued rel32, opening each run of adjacent pages once.
    fn flush_patches(&mut self) {
        if self.patches.is_empty() {
            return;
        }
        let mut patches = std::mem::take(&mut self.patches);
        patches.sort_unstable_by_key(|p| p.0);
        let page = self.page;
        let mut i = 0;
        while i < patches.len() {
            let lo = patches[i].0 & !(page - 1);
            let mut hi = (patches[i].0 + SITE_LEN + page - 1) & !(page - 1);
            let mut j = i + 1;
            while j < patches.len() && patches[j].0 & !(page - 1) <= hi {
                hi = hi.max((patches[j].0 + SITE_LEN + page - 1) & !(page - 1));
                j += 1;
            }
            let group = &patches[i..j];
            let res = self.with_writable(lo, hi - lo, || {
                for &(site, rel) in group {
                    unsafe {
                        debug_assert_eq!(*(site as *const u8), 0xE9);
                        core::ptr::write_unaligned((site + SITE_REL_OFFSET) as *mut i32, rel);
                    }
                }
            });
            // A link left unpatched or a stale link left in place cannot be
            // recovered from, like an allocation failure.
            if let Err(e) = res {
                panic!("rustarmic: code cache patch failed: {e:?}");
            }
            i = j;
        }
        patches.clear();
        self.patches = patches;
    }

    /// Publishes one block covering `[guest_pc, guest_end)` without linking
    /// its chain sites.  `cache_block` false installs a transient block that
    /// is never looked up.
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
        self.install_emitted(
            guest_pc,
            guest_end,
            guest_insns,
            bytes,
            chains,
            body_offset,
            cache_block,
            ThunkKind::Full,
        )
    }

    pub(crate) fn install_emitted(
        &mut self,
        guest_pc: u64,
        guest_end: u64,
        guest_insns: u32,
        bytes: &[u8],
        _chains: &[ChainSite],
        _body_offset: u32,
        cache_block: bool,
        thunk_kind: ThunkKind,
    ) -> Result<*const u8> {
        if !cache_block {
            return self.append_raw(bytes);
        }
        let end = guest_end.max(guest_pc.saturating_add(4));
        self.install_block(
            guest_pc,
            &[(guest_pc, end)],
            guest_insns,
            bytes,
            &[],
            None,
            thunk_kind,
        )
    }

    /// Publishes a translated block.  Cached blocks become dispatch targets
    /// and link with their neighbours; capped blocks are only reusable for
    /// the same (pc, cap) pair.
    pub(crate) fn install_block(
        &mut self,
        guest_pc: u64,
        ranges: &[(u64, u64)],
        guest_insns: u32,
        bytes: &[u8],
        chains: &[ChainSite],
        cap: Option<u32>,
        thunk_kind: ThunkKind,
    ) -> Result<*const u8> {
        let host_ptr = self.append_raw(bytes)?;
        let ranges: SmallVec<[(u64, u64); 4]> = if ranges.is_empty() {
            smallvec::smallvec![(guest_pc, guest_pc.saturating_add(4))]
        } else {
            ranges
                .iter()
                .map(|&(s, e)| if e <= s { (s, u64::MAX) } else { (s, e) })
                .collect()
        };
        if let Some(cap) = cap {
            self.capped.insert((guest_pc, cap), Capped { host_ptr, ranges });
            return Ok(host_ptr);
        }
        if self.table.contains_key(&guest_pc) {
            self.remove_block(guest_pc);
        }
        let mut out_sites = SmallVec::new();
        for c in chains {
            let site = host_ptr as usize + c.patch_offset as usize;
            out_sites.push((c.target_pc, site));
            self.links.entry(c.target_pc).or_default().push(site);
            if let Some(target) = self.table.get(&c.target_pc).map(|e| e.host_ptr) {
                self.queue_link(site, target);
            }
        }
        let guest_insns = guest_insns.max(1);
        self.table.insert(
            guest_pc,
            Entry {
                host_ptr,
                ranges,
                out_sites,
                guest_insns,
                thunk_kind,
            },
        );
        let index = dispatch_index(guest_pc);
        self.dispatch[index].set(DispatchSlot {
            pc: guest_pc,
            host: host_ptr as u64,
        });
        self.dispatch_meta[index].set((guest_insns, thunk_kind));
        if let Some(sites) = self.links.get(&guest_pc) {
            let sites: SmallVec<[usize; 8]> = sites.iter().copied().collect();
            for site in sites {
                self.queue_link(site, host_ptr);
            }
        }
        self.flush_patches();
        Ok(host_ptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protection(cache: &CodeCache, offset: usize) -> Protection {
        let ptr = unsafe { cache.memory.base().add(offset) };
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
        let sentinel = unsafe { cache.memory.base().add(page * 3) };
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
    fn rwx_publication_and_linking_keep_pages_writable_and_executable() {
        let page = region::page::size();
        let mut cache = CodeCache::with_rwx(page * 3, true).unwrap();
        let first = cache
            .install(0x1000, 0x1004, 1, &return_value(50), &[], 0, true)
            .unwrap();
        let transient = cache.append_raw(&return_value(51)).unwrap();
        assert_eq!(unsafe { call(&cache, first) }, 50);
        assert_eq!(unsafe { call(&cache, transient) }, 51);
        assert_eq!(protection(&cache, 0), Protection::READ_WRITE_EXECUTE);
        assert_eq!(protection(&cache, page), Protection::READ_WRITE_EXECUTE);
        cache.invalidate_range(0x1000, 4);
        assert!(cache.lookup(0x1000).is_none());
        cache.reset().unwrap();
        let replacement = cache
            .install(0x1000, 0x1004, 1, &return_value(52), &[], 0, true)
            .unwrap();
        assert_eq!(unsafe { call(&cache, replacement) }, 52);
        assert_eq!(protection(&cache, 0), Protection::READ_WRITE_EXECUTE);
    }

    #[cfg(windows)]
    #[test]
    fn rwx_cache_commits_only_the_pages_it_publishes() {
        let mut cache = CodeCache::with_rwx(1 << 30, true).unwrap();
        let block = cache
            .install(0x1000, 0x1004, 1, &return_value(53), &[], 0, true)
            .unwrap();
        assert_eq!(unsafe { call(&cache, block) }, 53);
        assert_eq!(protection(&cache, 0), Protection::READ_WRITE_EXECUTE);
        let far = unsafe { cache.memory.base().add(512 << 20) };
        assert!(!region::query(far).unwrap().is_committed());
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
        assert_eq!(cache.links.len(), 0);
        let replacement = cache
            .install(0x1000, 0x1004, 1, &return_value(49), &[], 0, true)
            .unwrap();
        assert_eq!(unsafe { call(&cache, replacement) }, 49);
        assert_eq!(protection(&cache, 0), Protection::READ_EXECUTE);
        assert_eq!(protection(&cache, page), Protection::READ_WRITE);
    }
}
