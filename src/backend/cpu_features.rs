use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuFeatures {
    pub has_gfni: bool,
}

static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

pub fn cpu_features() -> &'static CpuFeatures {
    FEATURES.get_or_init(detect)
}

#[cfg(target_arch = "x86_64")]
fn detect() -> CpuFeatures {
    let r = unsafe { std::arch::x86_64::__cpuid_count(7, 0) };
    CpuFeatures {
        has_gfni: (r.ecx & (1 << 8)) != 0,
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn detect() -> CpuFeatures { CpuFeatures::default() }
