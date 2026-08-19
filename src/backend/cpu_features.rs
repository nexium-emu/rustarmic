use std::cell::Cell;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuFeatures {
    pub has_sse41: bool,
    pub has_sse42: bool,
    pub has_ssse3: bool,
    pub has_avx: bool,
    pub has_fma: bool,
    pub has_gfni: bool,
    pub has_lzcnt: bool,
}

static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

thread_local! {
    static FEATURE_OVERRIDE: Cell<Option<CpuFeatures>> = const { Cell::new(None) };
}

pub fn cpu_features() -> &'static CpuFeatures {
    FEATURES.get_or_init(detect)
}

pub fn detect_features() -> CpuFeatures {
    detect()
}

pub fn active_features() -> CpuFeatures {
    FEATURE_OVERRIDE
        .with(|slot| slot.get())
        .unwrap_or(*cpu_features())
}

pub fn with_features<T>(features: CpuFeatures, f: impl FnOnce() -> T) -> T {
    FEATURE_OVERRIDE.with(|slot| {
        let previous = slot.replace(Some(features));
        let result = f();
        slot.set(previous);
        result
    })
}

#[cfg(target_arch = "x86_64")]
fn detect() -> CpuFeatures {
    let leaf7 = std::arch::x86_64::__cpuid_count(7, 0);
    let leaf_ext = std::arch::x86_64::__cpuid(0x80000001);
    CpuFeatures {
        has_sse41: std::arch::is_x86_feature_detected!("sse4.1"),
        has_sse42: std::arch::is_x86_feature_detected!("sse4.2"),
        has_ssse3: std::arch::is_x86_feature_detected!("ssse3"),
        has_avx: std::arch::is_x86_feature_detected!("avx"),
        has_fma: std::arch::is_x86_feature_detected!("fma"),
        has_gfni: (leaf7.ecx & (1 << 8)) != 0,
        has_lzcnt: (leaf_ext.ecx & (1 << 5)) != 0,
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn detect() -> CpuFeatures {
    CpuFeatures::default()
}
