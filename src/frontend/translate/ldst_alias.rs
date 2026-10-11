//! Load / store classes that behave exactly like another class at EL0.

use disarm64::decoder::{self, Operation};

use crate::error::{Error, Result};
use crate::frontend::translate;
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;

fn unsupported(em: &IrEmitter<'_>, raw: u32) -> Error {
    Error::Unsupported {
        pc: em.current_pc,
        opcode: raw,
    }
}

/// LDNP / STNP: LDP / STP (signed offset) with a non-temporal hint.  The
/// encodings differ only in bits 25:23 (000 vs 010).
pub fn translate_nap(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let pair = (raw & !(0b111 << 23)) | (0b010 << 23);
    match decoder::decode(pair).map(|o| o.operation) {
        Some(Operation::LDSTPAIR_OFF(i)) => translate::ldstpair_off::translate(em, i),
        _ => Err(unsupported(em, raw)),
    }
}

/// LDTR* / STTR*: unprivileged accesses, which at EL0 are the unscaled
/// LDUR* / STUR* forms (bits 11:10 = 10 instead of 00).
pub fn translate_unpriv(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let unscaled = raw & !(0b11 << 10);
    match decoder::decode(unscaled).map(|o| o.operation) {
        Some(Operation::LDST_UNSCALED(i)) => translate::ldst_unscaled::translate(em, i),
        _ => Err(unsupported(em, raw)),
    }
}
