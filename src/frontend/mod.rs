//! AArch64 → Armlet translation.

pub mod translator;
pub mod translate;

pub use translator::{translate_block_into, TranslateOptions};
