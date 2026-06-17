use disarm64::decoder::{ASIMDEXT, ASIMDPERM};

use crate::error::Result;
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

pub fn translate_ext(em: &mut IrEmitter<'_>, insn: ASIMDEXT) -> Result<InstStatus> {
    use ASIMDEXT::*;
    let raw = match insn { EXT_Vd_Vn_Vm_IDX(i) => i.0 };

    let q     = bit(raw, 30) == 1;
    let rm    = bits(raw, 16, 5) as u8;
    let imm4  = bits(raw, 11, 4);
    let rn    = bits(raw, 5, 5) as u8;
    let rd    = bits(raw, 0, 5) as u8;
    let byte_off = if q { imm4 } else { imm4 & 0x7 };

    let vn = em.get_v_q(rn);
    let vm = em.get_v_q(rm);
    let result = em.vec_ext(vn, vm, byte_off, q);
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}

#[derive(Clone, Copy)]
enum PermKind { Zip1, Zip2, Uzp1, Uzp2, Trn1, Trn2 }

pub fn translate_perm(em: &mut IrEmitter<'_>, insn: ASIMDPERM) -> Result<InstStatus> {
    use ASIMDPERM::*;
    let (raw, kind) = match insn {
        ZIP1_Vd_Vn_Vm(i) => (i.0, PermKind::Zip1),
        ZIP2_Vd_Vn_Vm(i) => (i.0, PermKind::Zip2),
        UZP1_Vd_Vn_Vm(i) => (i.0, PermKind::Uzp1),
        UZP2_Vd_Vn_Vm(i) => (i.0, PermKind::Uzp2),
        TRN1_Vd_Vn_Vm(i) => (i.0, PermKind::Trn1),
        TRN2_Vd_Vn_Vm(i) => (i.0, PermKind::Trn2),
    };

    let q    = bit(raw, 30) == 1;
    let size = bits(raw, 22, 2);
    let rm   = bits(raw, 16, 5) as u8;
    let rn   = bits(raw, 5,  5) as u8;
    let rd   = bits(raw, 0,  5) as u8;

    let vn = em.get_v_q(rn);
    let vm = em.get_v_q(rm);
    let result = match kind {
        PermKind::Zip1 => em.vec_zip1(vn, vm, size, q),
        PermKind::Zip2 => em.vec_zip2(vn, vm, size, q),
        PermKind::Uzp1 => em.vec_uzp1(vn, vm, size, q),
        PermKind::Uzp2 => em.vec_uzp2(vn, vm, size, q),
        PermKind::Trn1 => em.vec_trn1(vn, vm, size, q),
        PermKind::Trn2 => em.vec_trn2(vn, vm, size, q),
    };
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
