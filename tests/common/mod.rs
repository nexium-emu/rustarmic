//! Shared test-side module kept around so tests that don't touch memory
//! can `mod common;` without churn. After the M1.4 fn-ptr migration the
//! panicking memory handlers live on `CpuContext::default()` itself, so
//! this file no longer needs to provide `#[no_mangle]` linker symbols.
