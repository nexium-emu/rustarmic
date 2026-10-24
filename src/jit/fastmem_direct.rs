use core::cell::Cell;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::backend::DirectSite;

static HANDLED: AtomicU64 = AtomicU64::new(0);

pub fn handled_faults() -> u64 {
    HANDLED.load(Ordering::Relaxed)
}

const PAGE: usize = 4096;
const SHORT_JMP: u8 = 0xEB;
const NOP: u8 = 0x90;

#[derive(Clone, Copy)]
#[repr(C)]
struct Site {
    offset: u32,
    slow: u32,
    len: u8,
}

pub struct SiteRegistry {
    base: usize,
    rwx: bool,
    sites: Vec<Site>,
    patched: AtomicU64,
}

thread_local! {
    static CURRENT: Cell<*const SiteRegistry> = const { Cell::new(core::ptr::null()) };
}

impl SiteRegistry {
    pub(crate) fn new(base: usize, rwx: bool) -> Self {
        Self {
            base,
            rwx,
            sites: Vec::new(),
            patched: AtomicU64::new(0),
        }
    }

    pub(crate) fn register(&mut self, host: usize, sites: &[DirectSite]) {
        let block = host - self.base;
        for site in sites {
            let entry = Site {
                offset: (block + site.offset as usize) as u32,
                slow: (block + site.slow_offset as usize) as u32,
                len: site.len,
            };
            match self.sites.last() {
                Some(last) if last.offset >= entry.offset => {
                    let at = self.sites.partition_point(|s| s.offset < entry.offset);
                    self.sites.insert(at, entry);
                }
                _ => self.sites.push(entry),
            }
        }
    }

    pub(crate) fn clear(&mut self) {
        self.sites.clear();
    }

    pub fn len(&self) -> usize {
        self.sites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    pub fn patched(&self) -> u64 {
        self.patched.load(Ordering::Relaxed)
    }

    fn find(&self, rip: usize) -> Option<Site> {
        let offset = u32::try_from(rip.checked_sub(self.base)?).ok()?;
        let sites = &self.sites;
        let index = sites.binary_search_by_key(&offset, |s| s.offset).ok()?;
        Some(sites[index])
    }

    unsafe fn divert(&self, rip: usize) -> Option<usize> {
        let site = self.find(rip)?;
        HANDLED.fetch_add(1, Ordering::Relaxed);
        let slow = self.base + site.slow as usize;
        let rel = slow as isize - (rip as isize + 2);
        if let Ok(rel) = i8::try_from(rel) {
            let patched = if self.rwx {
                unsafe { write_patch(rip, site.len, rel) };
                true
            } else {
                unsafe { os::with_writable(rip, site.len as usize, || write_patch(rip, site.len, rel)) }
            };
            if patched {
                self.patched.fetch_add(1, Ordering::Relaxed);
                return Some(rip);
            }
        }
        Some(slow)
    }
}

impl Drop for SiteRegistry {
    fn drop(&mut self) {
        let me: *const SiteRegistry = self;
        let _ = CURRENT.try_with(|current| {
            if current.get() == me {
                current.set(core::ptr::null());
            }
        });
    }
}

unsafe fn write_patch(site: usize, len: u8, rel: i8) {
    unsafe {
        for i in 2..len as usize {
            core::ptr::write_volatile((site + i) as *mut u8, NOP);
        }
        core::ptr::write_volatile((site + 1) as *mut u8, rel as u8);
        core::ptr::write_volatile(site as *mut u8, SHORT_JMP);
    }
    core::sync::atomic::compiler_fence(Ordering::SeqCst);
}

fn page_span(addr: usize, len: usize) -> (usize, usize) {
    let lo = addr & !(PAGE - 1);
    let hi = (addr + len + PAGE - 1) & !(PAGE - 1);
    (lo, hi - lo)
}

pub(crate) struct Entered {
    prev: *const SiteRegistry,
}

impl Entered {
    pub(crate) fn new(registry: *const SiteRegistry) -> Self {
        let prev = CURRENT.replace(registry);
        Self { prev }
    }
}

impl Drop for Entered {
    fn drop(&mut self) {
        CURRENT.set(self.prev);
    }
}

unsafe fn divert_current(rip: usize) -> Option<usize> {
    let registry = CURRENT.get();
    if registry.is_null() {
        return None;
    }
    unsafe { (*registry).divert(rip) }
}

pub(crate) fn install_handler() -> bool {
    os::install()
}

#[cfg(windows)]
mod os {
    use core::ffi::c_void;
    use std::sync::Once;
    use std::sync::atomic::{AtomicBool, Ordering};

    use crate::jit::page_table::os::{AddVectoredExceptionHandler, ExceptionPointers};

    const STATUS_ACCESS_VIOLATION: u32 = 0xC000_0005;
    const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
    const PAGE_READWRITE: u32 = 0x04;

    #[repr(C)]
    struct Context {
        p_home: [u64; 6],
        context_flags: u32,
        mxcsr: u32,
        segments: [u16; 6],
        eflags: u32,
        debug: [u64; 6],
        rax: u64,
        rcx: u64,
        rdx: u64,
        rbx: u64,
        rsp: u64,
        rbp: u64,
        rsi: u64,
        rdi: u64,
        r8: u64,
        r9: u64,
        r10: u64,
        r11: u64,
        r12: u64,
        r13: u64,
        r14: u64,
        r15: u64,
        rip: u64,
    }

    const _: () = assert!(core::mem::offset_of!(Context, rip) == 0xF8);

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualProtect(addr: *mut c_void, size: usize, protect: u32, old: *mut u32) -> i32;
    }

    static INSTALL: Once = Once::new();
    static INSTALLED: AtomicBool = AtomicBool::new(false);

    unsafe extern "system" fn on_fault(pointers: *mut ExceptionPointers) -> i32 {
        unsafe {
            let Some(pointers) = pointers.as_ref() else {
                return EXCEPTION_CONTINUE_SEARCH;
            };
            let (Some(record), Some(context)) = (
                pointers.record.as_ref(),
                pointers.context.cast::<Context>().as_mut(),
            ) else {
                return EXCEPTION_CONTINUE_SEARCH;
            };
            if record.code != STATUS_ACCESS_VIOLATION {
                return EXCEPTION_CONTINUE_SEARCH;
            }
            match super::divert_current(context.rip as usize) {
                Some(resume) => {
                    context.rip = resume as u64;
                    EXCEPTION_CONTINUE_EXECUTION
                }
                None => EXCEPTION_CONTINUE_SEARCH,
            }
        }
    }

    pub fn install() -> bool {
        INSTALL.call_once(|| {
            let handle = unsafe { AddVectoredExceptionHandler(1, on_fault) };
            INSTALLED.store(!handle.is_null(), Ordering::Release);
        });
        INSTALLED.load(Ordering::Acquire)
    }

    pub unsafe fn with_writable(addr: usize, len: usize, write: impl FnOnce()) -> bool {
        let (lo, span) = super::page_span(addr, len);
        let mut old = 0u32;
        unsafe {
            if VirtualProtect(lo as *mut c_void, span, PAGE_READWRITE, &mut old) == 0 {
                return false;
            }
            write();
            let mut restored = 0u32;
            VirtualProtect(lo as *mut c_void, span, old, &mut restored) != 0
        }
    }
}

#[cfg(all(unix, target_arch = "x86_64"))]
mod os {
    use core::ffi::c_void;
    use std::sync::Once;
    use std::sync::atomic::{AtomicBool, Ordering};

    const SIGSEGV: i32 = 11;
    #[cfg(target_os = "linux")]
    const SIGBUS: i32 = 7;
    #[cfg(not(target_os = "linux"))]
    const SIGBUS: i32 = 10;

    #[cfg(target_os = "linux")]
    const SA_SIGINFO: i32 = 0x4;
    #[cfg(target_os = "linux")]
    const SA_ONSTACK: i32 = 0x0800_0000;
    #[cfg(target_os = "linux")]
    const SA_NODEFER: i32 = 0x4000_0000;
    #[cfg(not(target_os = "linux"))]
    const SA_ONSTACK: i32 = 0x1;
    #[cfg(not(target_os = "linux"))]
    const SA_NODEFER: i32 = 0x10;
    #[cfg(not(target_os = "linux"))]
    const SA_SIGINFO: i32 = 0x40;

    const SIG_DFL: usize = 0;
    const SIG_IGN: usize = 1;
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const PROT_EXEC: i32 = 4;

    #[cfg(target_os = "linux")]
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SigAction {
        handler: usize,
        mask: [u64; 16],
        flags: i32,
        restorer: usize,
    }

    #[cfg(not(target_os = "linux"))]
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct SigAction {
        handler: usize,
        mask: u32,
        flags: i32,
    }

    #[cfg(target_os = "linux")]
    const _: () = assert!(core::mem::size_of::<SigAction>() == 152);
    #[cfg(not(target_os = "linux"))]
    const _: () = assert!(core::mem::size_of::<SigAction>() == 16);

    impl SigAction {
        const fn empty() -> Self {
            #[cfg(target_os = "linux")]
            {
                Self {
                    handler: 0,
                    mask: [0; 16],
                    flags: 0,
                    restorer: 0,
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                Self {
                    handler: 0,
                    mask: 0,
                    flags: 0,
                }
            }
        }
    }

    type SigInfoHandler = unsafe extern "C" fn(i32, *mut c_void, *mut c_void);
    type SigHandler = unsafe extern "C" fn(i32);

    unsafe extern "C" {
        fn sigaction(sig: i32, act: *const SigAction, old: *mut SigAction) -> i32;
        fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
    }

    static INSTALL: Once = Once::new();
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    static mut OLD_SEGV: SigAction = SigAction::empty();
    static mut OLD_BUS: SigAction = SigAction::empty();

    unsafe fn rip_slot(uctx: *mut c_void) -> *mut u64 {
        #[cfg(target_os = "linux")]
        unsafe {
            uctx.cast::<u8>().add(40 + 16 * 8).cast::<u64>()
        }
        #[cfg(not(target_os = "linux"))]
        unsafe {
            let mcontext = *uctx.cast::<u8>().add(48).cast::<*mut u8>();
            mcontext.add(16 + 16 * 8).cast::<u64>()
        }
    }

    unsafe fn chain(sig: i32, info: *mut c_void, uctx: *mut c_void) {
        let old = if sig == SIGBUS {
            unsafe { core::ptr::addr_of!(OLD_BUS).read() }
        } else {
            unsafe { core::ptr::addr_of!(OLD_SEGV).read() }
        };
        unsafe {
            if old.flags & SA_SIGINFO != 0 && old.handler > SIG_IGN {
                let handler: SigInfoHandler = core::mem::transmute(old.handler);
                handler(sig, info, uctx);
            } else if old.handler == SIG_DFL {
                sigaction(sig, &old, core::ptr::null_mut());
            } else if old.handler != SIG_IGN {
                let handler: SigHandler = core::mem::transmute(old.handler);
                handler(sig);
            }
        }
    }

    unsafe extern "C" fn on_fault(sig: i32, info: *mut c_void, uctx: *mut c_void) {
        unsafe {
            if !uctx.is_null() {
                let slot = rip_slot(uctx);
                if let Some(resume) = super::divert_current(*slot as usize) {
                    *slot = resume as u64;
                    return;
                }
            }
            chain(sig, info, uctx);
        }
    }

    unsafe fn install_one(sig: i32, old: *mut SigAction) -> bool {
        let mut act = SigAction::empty();
        act.handler = on_fault as SigInfoHandler as usize;
        act.flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
        unsafe { sigaction(sig, &act, old) == 0 }
    }

    pub fn install() -> bool {
        INSTALL.call_once(|| {
            let ok = unsafe {
                install_one(SIGSEGV, core::ptr::addr_of_mut!(OLD_SEGV))
                    && install_one(SIGBUS, core::ptr::addr_of_mut!(OLD_BUS))
            };
            INSTALLED.store(ok, Ordering::Release);
        });
        INSTALLED.load(Ordering::Acquire)
    }

    pub unsafe fn with_writable(addr: usize, len: usize, write: impl FnOnce()) -> bool {
        let (lo, span) = super::page_span(addr, len);
        unsafe {
            if mprotect(lo as *mut c_void, span, PROT_READ | PROT_WRITE) != 0 {
                return false;
            }
            write();
            mprotect(lo as *mut c_void, span, PROT_READ | PROT_EXEC) == 0
        }
    }
}

#[cfg(not(any(windows, all(unix, target_arch = "x86_64"))))]
mod os {
    pub fn install() -> bool {
        false
    }

    pub unsafe fn with_writable(_addr: usize, _len: usize, _write: impl FnOnce()) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site(offset: u32, len: u8, slow_offset: u32) -> DirectSite {
        DirectSite {
            offset,
            len,
            slow_offset,
        }
    }

    #[test]
    fn registry_finds_registered_sites_and_forgets_them_on_clear() {
        let mut registry = SiteRegistry::new(0x1000, true);
        registry.register(0x1100, &[site(4, 3, 12), site(40, 5, 60)]);
        registry.register(0x1200, &[site(8, 4, 20)]);
        registry.register(0x1080, &[site(0, 3, 9)]);
        let offsets: Vec<u32> = registry.sites.iter().map(|s| s.offset).collect();
        assert_eq!(offsets, vec![0x80, 0x104, 0x128, 0x208]);
        assert_eq!(registry.find(0x1104).map(|s| s.len), Some(3));
        assert_eq!(registry.find(0x1104).map(|s| s.slow), Some(0x10C));
        assert_eq!(registry.find(0x1208).map(|s| s.len), Some(4));
        assert!(registry.find(0x1105).is_none());
        assert!(registry.find(0x0FFF).is_none());
        registry.clear();
        assert!(registry.is_empty());
        assert!(registry.find(0x1104).is_none());
    }

    #[test]
    fn patch_writes_short_jump_padded_with_nops() {
        let mut code = [0x48u8, 0x8B, 0x04, 0x10, 0xEB, 0x05, 0xCC, 0xCC];
        let site = code.as_mut_ptr() as usize;
        unsafe { write_patch(site, 4, 4) };
        assert_eq!(code, [0xEB, 0x04, 0x90, 0x90, 0xEB, 0x05, 0xCC, 0xCC]);
    }

    #[test]
    fn entered_publishes_and_restores_the_registry_pointer() {
        let outer = SiteRegistry::new(0, true);
        let inner = SiteRegistry::new(0, true);
        assert!(CURRENT.get().is_null());
        {
            let _outer = Entered::new(&outer);
            assert_eq!(CURRENT.get(), &outer as *const _);
            {
                let _inner = Entered::new(&inner);
                assert_eq!(CURRENT.get(), &inner as *const _);
            }
            assert_eq!(CURRENT.get(), &outer as *const _);
        }
        assert!(CURRENT.get().is_null());
    }

    #[test]
    fn page_span_covers_an_instruction_crossing_a_page() {
        assert_eq!(page_span(0x1FFE, 4), (0x1000, 0x2000));
        assert_eq!(page_span(0x2000, 3), (0x2000, 0x1000));
    }
}
