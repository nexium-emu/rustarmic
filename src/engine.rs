//! Safe, embeddable Rustarmic surface.
//!
//! The original `Jit` type remains available for low-level tests and legacy
//! adapters.  New users should share an `Engine` (and its `SharedRuntime`)
//! instead of passing raw host pointers around.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use thiserror::Error;

use crate::{CpuContext, CpuFeatures, Error, Jit, JitConfig, Memory, RunOutcome};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FpMode {
    #[default]
    Accurate,
    Fast,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemoryMode {
    #[default]
    PageTable,
    Fastmem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuestIsaFeatures {
    pub atomics: bool,
    pub crc: bool,
    pub aes: bool,
    pub sha1: bool,
    pub sha2: bool,
    pub fp_asimd: bool,
}

impl Default for GuestIsaFeatures {
    fn default() -> Self {
        Self {
            atomics: true,
            crc: true,
            aes: true,
            sha1: true,
            sha2: true,
            fp_asimd: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub guest_isa: GuestIsaFeatures,
    pub host_features: CpuFeatures,
    pub fp_mode: FpMode,
    pub memory_mode: MemoryMode,
    pub code_cache_bytes: usize,
    pub max_block_insts: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        let host_features = crate::backend::cpu_features::detect_features();
        Self {
            guest_isa: GuestIsaFeatures::default(),
            host_features,
            fp_mode: FpMode::Accurate,
            memory_mode: MemoryMode::PageTable,
            code_cache_bytes: 256 * 1024 * 1024,
            max_block_insts: 64,
        }
    }
}

#[derive(Debug, Error)]
pub enum CpuError {
    #[error("host allocation failed: {0}")]
    HostAllocation(String),
    #[error("code emission failed: {0}")]
    Emission(String),
    #[error("corrupted internal state: {0}")]
    CorruptState(String),
    #[error("guest execution failed: {0}")]
    Guest(String),
}

impl From<Error> for CpuError {
    fn from(error: Error) -> Self {
        match error {
            Error::HostAlloc(message) => Self::HostAllocation(message),
            Error::CodeCacheFull => Self::HostAllocation("code cache exhausted".to_string()),
            Error::Backend(message) => Self::Emission(message),
            Error::UnsupportedHost => Self::HostAllocation(error.to_string()),
            other => Self::Guest(other.to_string()),
        }
    }
}

/// A generation-tagged mapping lease.  The lease is intentionally opaque:
/// callers can retain it while a mapping is being replaced, but cannot obtain
/// a dangling host pointer from the public API.
#[derive(Clone)]
pub struct MappingLease {
    generation: u64,
    guest_base: u64,
    len: u64,
    backing: Arc<Vec<u8>>,
}

impl MappingLease {
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn guest_base(&self) -> u64 {
        self.guest_base
    }
    pub fn len(&self) -> u64 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn bytes(&self) -> &[u8] {
        &self.backing
    }
}

pub struct SharedRuntime {
    generation: AtomicU64,
    clock_start: Instant,
    mappings: RwLock<Vec<MappingLease>>,
    jit: Mutex<Jit>,
}

impl SharedRuntime {
    fn new(config: &EngineConfig) -> Result<Self, CpuError> {
        if config.max_block_insts == 0 || config.max_block_insts > 64 {
            return Err(CpuError::CorruptState(
                "max_block_insts must be in 1..=64".to_string(),
            ));
        }
        let jit_config = JitConfig {
            code_cache_bytes: config.code_cache_bytes,
            use_fastmem: matches!(config.memory_mode, MemoryMode::Fastmem),
            host_features: Some(config.host_features),
            translate: crate::frontend::TranslateOptions {
                max_insts: config.max_block_insts,
                ..Default::default()
            },
        };
        let jit = Jit::new(jit_config).map_err(CpuError::from)?;
        Ok(Self {
            generation: AtomicU64::new(0),
            clock_start: Instant::now(),
            mappings: RwLock::new(Vec::new()),
            jit: Mutex::new(jit),
        })
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Guest counter in the architecturally exposed 19.2 MHz domain.
    pub fn guest_ticks(&self) -> u64 {
        let ticks = self
            .clock_start
            .elapsed()
            .as_nanos()
            .saturating_mul(19_200_000)
            / 1_000_000_000;
        ticks.min(u128::from(u64::MAX)) as u64
    }

    pub fn map_owned(&self, guest_base: u64, bytes: Vec<u8>) -> Result<MappingLease, CpuError> {
        let len = u64::try_from(bytes.len())
            .map_err(|_| CpuError::CorruptState("mapping length overflow".to_string()))?;
        guest_base
            .checked_add(len)
            .ok_or_else(|| CpuError::CorruptState("mapping address overflow".to_string()))?;
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        let lease = MappingLease {
            generation,
            guest_base,
            len,
            backing: Arc::new(bytes),
        };
        self.mappings.write().unwrap().push(lease.clone());
        Ok(lease)
    }

    pub fn unmap(&self, guest_base: u64, len: u64) -> Result<(), CpuError> {
        let end = guest_base
            .checked_add(len)
            .ok_or_else(|| CpuError::CorruptState("unmapping address overflow".to_string()))?;
        self.mappings.write().unwrap().retain(|mapping| {
            mapping.guest_base >= end || mapping.guest_base + mapping.len <= guest_base
        });
        self.generation.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}

#[derive(Clone)]
pub struct Engine {
    runtime: Arc<SharedRuntime>,
}

impl Engine {
    pub fn new(config: EngineConfig) -> Result<Self, CpuError> {
        Ok(Self {
            runtime: Arc::new(SharedRuntime::new(&config)?),
        })
    }

    pub fn shared_runtime(&self) -> Arc<SharedRuntime> {
        Arc::clone(&self.runtime)
    }

    pub fn run(
        &self,
        ctx: &mut CpuContext,
        memory: &mut dyn Memory,
        budget: u64,
    ) -> Result<RunOutcome, CpuError> {
        self.runtime
            .jit
            .lock()
            .map_err(|_| CpuError::CorruptState("JIT mutex poisoned".to_string()))?
            .run_bounded(ctx, memory, budget)
            .map_err(CpuError::from)
    }

    pub fn step(
        &self,
        ctx: &mut CpuContext,
        memory: &mut dyn Memory,
    ) -> Result<RunOutcome, CpuError> {
        self.run(ctx, memory, 1)
    }

    pub fn invalidate_range(&self, start: u64, len: u64) -> Result<(), CpuError> {
        let mut jit = self
            .runtime
            .jit
            .lock()
            .map_err(|_| CpuError::CorruptState("JIT mutex poisoned".to_string()))?;
        jit.invalidate_range(start, len);
        Ok(())
    }
}
