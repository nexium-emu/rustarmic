use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuFeatures {
    pub has_gfni: bool,
    pub has_lzcnt: bool,
}

static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

pub fn cpu_features() -> &'static CpuFeatures {
    FEATURES.get_or_init(detect)
}

#[cfg(target_arch = "x86_64")]
fn detect() -> CpuFeatures {
    let leaf7 = std::arch::x86_64::__cpuid_count(7, 0);
    let leaf_ext = std::arch::x86_64::__cpuid(0x80000001);
    CpuFeatures {
        has_gfni:  (leaf7.ecx    & (1 << 8)) != 0,
        has_lzcnt: (leaf_ext.ecx & (1 << 5)) != 0,
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn detect() -> CpuFeatures { CpuFeatures::default() }
