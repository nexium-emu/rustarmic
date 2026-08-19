use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("decode failed at guest PC {pc:#x}: opcode {opcode:#010x}")]
    Decode { pc: u64, opcode: u32 },

    #[error("unsupported instruction at PC {pc:#x}: opcode {opcode:#010x}")]
    Unsupported { pc: u64, opcode: u32 },

    #[error("guest memory access failed at {addr:#x}")]
    GuestMemory { addr: u64 },

    #[error("code cache exhausted")]
    CodeCacheFull,

    #[error("backend emission failed: {0}")]
    Backend(String),

    #[error("host memory allocation failed: {0}")]
    HostAlloc(String),

    #[error("unsupported host CPU: Rustarmic requires x86-64 SSE4.1")]
    UnsupportedHost,

    #[error("translation block too large at PC {pc:#x}")]
    BlockTooLarge { pc: u64 },
}

pub type Result<T> = core::result::Result<T, Error>;

impl From<iced_x86::IcedError> for Error {
    #[inline]
    fn from(e: iced_x86::IcedError) -> Self {
        Error::Backend(e.to_string())
    }
}
