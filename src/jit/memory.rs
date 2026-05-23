//! Embedder memory interface.

/// Implemented by the host to supply guest memory.
///
/// `fetch_inst` is used by the translator; the JITted code itself reads/writes
/// memory either through `mem_base` (fastmem) or — in a follow-up — through
/// thunks back into the trait. The initial release uses the `mem_base`
/// fastmem path exclusively.
pub trait Memory {
    /// Fetch a 32-bit guest instruction word.
    fn fetch_inst(&mut self, addr: u64) -> Option<u32>;
}

/// A trivial in-memory implementation backed by a flat byte vector. Useful
/// for tests and small embedders.
pub struct FlatMemory {
    pub bytes: Vec<u8>,
    pub base:  u64,
}

impl FlatMemory {
    pub fn new(base: u64, size: usize) -> Self {
        Self { bytes: vec![0; size], base }
    }

    pub fn write(&mut self, addr: u64, data: &[u8]) {
        let off = (addr - self.base) as usize;
        self.bytes[off..off + data.len()].copy_from_slice(data);
    }

    pub fn write_u32(&mut self, addr: u64, value: u32) {
        let off = (addr - self.base) as usize;
        self.bytes[off..off + 4].copy_from_slice(&value.to_le_bytes());
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        // Note: returning a pointer that's relative to `base==0` semantics —
        // the caller wires this into `CpuContext::mem_base` only when
        // `base == 0`; otherwise the guest address space must be offset.
        self.bytes.as_mut_ptr()
    }
}

impl Memory for FlatMemory {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let off = addr.checked_sub(self.base)? as usize;
        if off + 4 > self.bytes.len() { return None; }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.bytes[off..off + 4]);
        Some(u32::from_le_bytes(buf))
    }
}
