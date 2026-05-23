//! Top-level AArch64 instruction classifier.
//!
//! We use the AArch64 top-level "op0" encoding split (ARMv8 ARM C4.1) to
//! route each 32-bit instruction word to a translator. This is intentionally
//! a hand-written switch rather than a generated decision tree — for the
//! initial coverage it's faster to read and faster to extend than codegen
//! from XML, and the match arms compile down to a dense jump table.

use crate::util::bits::bits;

/// Top-level instruction class. Mirrors the C4.1 "op0" decoding groups so
/// each translator only needs to inspect the bits relevant to its class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeClass {
    Reserved,
    Sme,            // 0001 — SME (ARMv9)
    Sve,            // 0010 — SVE
    DataProcImm,    // 100x
    BranchSysExc,   // 101x
    LoadStore,      // x1x0
    DataProcReg,    // x101
    DataProcFloat,  // x111
}

#[inline]
pub fn classify(inst: u32) -> DecodeClass {
    // op0 = inst[28:25]
    let op0 = bits(inst, 25, 4);
    match op0 {
        0b0000           => DecodeClass::Reserved,
        0b0001           => DecodeClass::Sme,
        0b0010           => DecodeClass::Sve,
        0b1000 | 0b1001  => DecodeClass::DataProcImm,
        0b1010 | 0b1011  => DecodeClass::BranchSysExc,
        // x1x0
        v if (v & 0b0101) == 0b0100 => DecodeClass::LoadStore,
        // x101
        v if (v & 0b0111) == 0b0101 => DecodeClass::DataProcReg,
        // x111
        v if (v & 0b0111) == 0b0111 => DecodeClass::DataProcFloat,
        _ => DecodeClass::Reserved,
    }
}
