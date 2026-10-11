//! Backing store for the fastmem page table (`CpuContext::page_table`).
//!
//! A table covering a 39-bit guest address space is 128 MiB, almost all of
//! it zero.  [`PageTableMem`] reserves the range and only pays for the pages
//! actually touched:
//!
//! * Linux: anonymous `MAP_NORESERVE` mapping; untouched pages read as the
//!   shared zero page and use neither memory nor commit charge.
//! * Windows: the range is only reserved (`MEM_RESERVE`, no commit charge).
//!   Pages are committed explicitly before the embedder writes them
//!   ([`PageTableMem::prepare_write`]) and, for reads by generated code of
//!   never-written pages, on demand by a vectored exception handler that
//!   commits the (zero-filled) page and resumes.
//! * Elsewhere: a zeroed heap allocation.
//!
//! The inline fastmem guard is unchanged: still one byte load per access.

use core::ops::{Deref, DerefMut};

pub struct PageTableMem {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for PageTableMem {}
unsafe impl Sync for PageTableMem {}

impl PageTableMem {
    /// Reserves a zero-filled table of `len` bytes.
    pub fn new(len: usize) -> Option<Self> {
        let len = len.max(1);
        let ptr = os::reserve(len)?;
        Some(Self { ptr, len })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Makes `range` writable without faulting.  Writing through the slice
    /// is correct without it; this only avoids one exception per page on
    /// platforms that commit on demand.
    pub fn prepare_write(&mut self, range: core::ops::Range<usize>) {
        let end = range.end.min(self.len);
        if range.start < end {
            os::commit(unsafe { self.ptr.add(range.start) }, end - range.start);
        }
    }
}

impl Deref for PageTableMem {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for PageTableMem {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for PageTableMem {
    fn drop(&mut self) {
        os::release(self.ptr, self.len);
    }
}

#[cfg(target_os = "linux")]
mod os {
    use core::ffi::c_void;

    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_PRIVATE: i32 = 0x02;
    const MAP_ANONYMOUS: i32 = 0x20;
    const MAP_NORESERVE: i32 = 0x4000;

    unsafe extern "C" {
        fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
    }

    pub fn reserve(len: usize) -> Option<*mut u8> {
        let p = unsafe {
            mmap(
                core::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                -1,
                0,
            )
        };
        (p as isize != -1 && !p.is_null()).then_some(p.cast())
    }

    pub fn commit(_ptr: *mut u8, _len: usize) {}

    pub fn release(ptr: *mut u8, len: usize) {
        unsafe { munmap(ptr.cast(), len) };
    }
}

#[cfg(windows)]
mod os {
    use core::ffi::c_void;
    use std::sync::Once;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_READWRITE: u32 = 0x04;
    const STATUS_ACCESS_VIOLATION: u32 = 0xC000_0005;
    const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
    const PAGE: usize = 4096;

    #[repr(C)]
    struct ExceptionRecord {
        code: u32,
        flags: u32,
        record: *mut ExceptionRecord,
        address: *mut c_void,
        parameters: u32,
        information: [usize; 15],
    }

    #[repr(C)]
    struct ExceptionPointers {
        record: *mut ExceptionRecord,
        context: *mut c_void,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
        fn VirtualFree(addr: *mut c_void, size: usize, kind: u32) -> i32;
        fn AddVectoredExceptionHandler(
            first: u32,
            handler: unsafe extern "system" fn(*mut ExceptionPointers) -> i32,
        ) -> *mut c_void;
    }

    /// Live reservations the exception handler may commit into.  Fixed
    /// slots so the handler never locks or allocates.
    const SLOTS: usize = 64;
    static BASE: [AtomicUsize; SLOTS] = [const { AtomicUsize::new(0) }; SLOTS];
    static LEN: [AtomicUsize; SLOTS] = [const { AtomicUsize::new(0) }; SLOTS];
    static HANDLER: Once = Once::new();

    unsafe extern "system" fn on_fault(pointers: *mut ExceptionPointers) -> i32 {
        unsafe {
            let Some(record) = pointers.as_ref().and_then(|p| p.record.as_ref()) else {
                return EXCEPTION_CONTINUE_SEARCH;
            };
            if record.code != STATUS_ACCESS_VIOLATION || record.parameters < 2 {
                return EXCEPTION_CONTINUE_SEARCH;
            }
            let addr = record.information[1];
            for i in 0..SLOTS {
                let base = BASE[i].load(Ordering::Acquire);
                let len = LEN[i].load(Ordering::Acquire);
                if base != 0 && addr >= base && addr < base + len {
                    let page = addr & !(PAGE - 1);
                    let p = VirtualAlloc(page as *mut c_void, PAGE, MEM_COMMIT, PAGE_READWRITE);
                    return if p.is_null() {
                        EXCEPTION_CONTINUE_SEARCH
                    } else {
                        EXCEPTION_CONTINUE_EXECUTION
                    };
                }
            }
            EXCEPTION_CONTINUE_SEARCH
        }
    }

    pub fn reserve(len: usize) -> Option<*mut u8> {
        HANDLER.call_once(|| unsafe {
            AddVectoredExceptionHandler(1, on_fault);
        });
        let p = unsafe { VirtualAlloc(core::ptr::null_mut(), len, MEM_RESERVE, PAGE_NOACCESS) };
        if p.is_null() {
            return None;
        }
        for i in 0..SLOTS {
            if BASE[i]
                .compare_exchange(0, usize::MAX, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                LEN[i].store(len, Ordering::Release);
                BASE[i].store(p as usize, Ordering::Release);
                return Some(p.cast());
            }
        }
        // No slot for on-demand commits: commit everything up front.
        unsafe { VirtualAlloc(p, len, MEM_COMMIT, PAGE_READWRITE) };
        Some(p.cast())
    }

    pub fn commit(ptr: *mut u8, len: usize) {
        let start = (ptr as usize) & !(PAGE - 1);
        let end = (ptr as usize + len + PAGE - 1) & !(PAGE - 1);
        unsafe { VirtualAlloc(start as *mut c_void, end - start, MEM_COMMIT, PAGE_READWRITE) };
    }

    pub fn release(ptr: *mut u8, _len: usize) {
        for i in 0..SLOTS {
            if BASE[i].load(Ordering::Acquire) == ptr as usize {
                BASE[i].store(usize::MAX, Ordering::Release);
                LEN[i].store(0, Ordering::Release);
                BASE[i].store(0, Ordering::Release);
            }
        }
        unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) };
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod os {
    use std::alloc::{Layout, alloc_zeroed, dealloc};

    fn layout(len: usize) -> Layout {
        Layout::from_size_align(len, 4096).unwrap()
    }

    pub fn reserve(len: usize) -> Option<*mut u8> {
        let p = unsafe { alloc_zeroed(layout(len)) };
        (!p.is_null()).then_some(p)
    }

    pub fn commit(_ptr: *mut u8, _len: usize) {}

    pub fn release(ptr: *mut u8, len: usize) {
        unsafe { dealloc(ptr, layout(len)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_table_reads_zero_and_costs_only_touched_pages() {
        // 128 MiB, as for a 39-bit guest.
        let mut t = PageTableMem::new(1 << 27).unwrap();
        assert_eq!(t[0], 0);
        assert_eq!(t[(1 << 27) - 1], 0);
        assert_eq!(t[12_345_678], 0);
        t.prepare_write(5000..9000);
        t[5000..9000].fill(3);
        t[100_000_000] = 1;
        assert_eq!(t[4999], 0);
        assert_eq!(t[5000], 3);
        assert_eq!(t[8999], 3);
        assert_eq!(t[100_000_000], 1);
        #[cfg(target_os = "linux")]
        {
            // Resident set must not include the whole table.
            let status = std::fs::read_to_string("/proc/self/status").unwrap();
            let rss_kb: u64 = status
                .lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
                .unwrap();
            assert!(rss_kb < 100 * 1024, "RSS {rss_kb} KiB");
        }
    }
}
