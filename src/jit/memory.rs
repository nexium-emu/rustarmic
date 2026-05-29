pub trait Memory {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32>;
}

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


