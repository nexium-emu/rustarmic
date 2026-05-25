//! ASIMD "different"-form widening ops. Coverage so far: SADDL/SADDL2 and
//! UADDL/UADDL2 (widening signed/unsigned add). SSUBL/USUBL/MULL etc. share
//! the same shape and can slot in here as needed.

use disarm64::decoder::ASIMDDIFF;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum Kind { Saddl, Uaddl, Ssubl, Usubl, Smull, Umull }

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDDIFF) -> Result<InstStatus> {
    use ASIMDDIFF::*;
    let (raw, kind) = match insn {
        SADDL_Vd_Vn_Vm(i)  => (i.0, Kind::Saddl),
        SADDL2_Vd_Vn_Vm(i) => (i.0, Kind::Saddl),
        UADDL_Vd_Vn_Vm(i)  => (i.0, Kind::Uaddl),
        UADDL2_Vd_Vn_Vm(i) => (i.0, Kind::Uaddl),
        SSUBL_Vd_Vn_Vm(i)  => (i.0, Kind::Ssubl),
        SSUBL2_Vd_Vn_Vm(i) => (i.0, Kind::Ssubl),
        USUBL_Vd_Vn_Vm(i)  => (i.0, Kind::Usubl),
        USUBL2_Vd_Vn_Vm(i) => (i.0, Kind::Usubl),
        SMULL_Vd_Vn_Vm(i)  => (i.0, Kind::Smull),
        SMULL2_Vd_Vn_Vm(i) => (i.0, Kind::Smull),
        UMULL_Vd_Vn_Vm(i)  => (i.0, Kind::Umull),
        UMULL2_Vd_Vn_Vm(i) => (i.0, Kind::Umull),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };
    translate_with(em, raw, kind)
}

fn translate_with(em: &mut IrEmitter<'_>, raw: u32, kind: Kind) -> Result<InstStatus> {
    let high_half = bit(raw, 30) == 1;
    let size = bits(raw, 22, 2);  // 00=B->H, 01=H->S, 10=S->D
    let rm   = bits(raw, 16, 5) as u8;
    let rn   = bits(raw, 5,  5) as u8;
    let rd   = bits(raw, 0,  5) as u8;
    if size > 2 {
        return Err(Error::Decode { pc: em.current_pc, opcode: raw });
    }
    let vn = em.get_v_q(rn);
    let vm = em.get_v_q(rm);
    let result = match kind {
        Kind::Saddl => em.vec_saddl(vn, vm, size, high_half),
        Kind::Uaddl => em.vec_uaddl(vn, vm, size, high_half),
        Kind::Ssubl => em.vec_ssubl(vn, vm, size, high_half),
        Kind::Usubl => em.vec_usubl(vn, vm, size, high_half),
        Kind::Smull => em.vec_smull(vn, vm, size, high_half),
        Kind::Umull => em.vec_umull(vn, vm, size, high_half),
    };
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
