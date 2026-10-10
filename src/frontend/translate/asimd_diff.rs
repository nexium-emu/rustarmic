use disarm64::decoder::ASIMDDIFF;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum Kind {
    Saddl,
    Uaddl,
    Ssubl,
    Usubl,
    Saddw,
    Uaddw,
    Ssubw,
    Usubw,
    Smull,
    Umull,
    Smlal,
    Umlal,
    Smlsl,
    Umlsl,
}

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDDIFF) -> Result<InstStatus> {
    use ASIMDDIFF::*;
    let (raw, kind) = match insn {
        SADDL_Vd_Vn_Vm(i) => (i.0, Kind::Saddl),
        SADDL2_Vd_Vn_Vm(i) => (i.0, Kind::Saddl),
        UADDL_Vd_Vn_Vm(i) => (i.0, Kind::Uaddl),
        UADDL2_Vd_Vn_Vm(i) => (i.0, Kind::Uaddl),
        SSUBL_Vd_Vn_Vm(i) => (i.0, Kind::Ssubl),
        SSUBL2_Vd_Vn_Vm(i) => (i.0, Kind::Ssubl),
        USUBL_Vd_Vn_Vm(i) => (i.0, Kind::Usubl),
        USUBL2_Vd_Vn_Vm(i) => (i.0, Kind::Usubl),
        SADDW_Vd_Vn_Vm(i) => (i.0, Kind::Saddw),
        SADDW2_Vd_Vn_Vm(i) => (i.0, Kind::Saddw),
        UADDW_Vd_Vn_Vm(i) => (i.0, Kind::Uaddw),
        UADDW2_Vd_Vn_Vm(i) => (i.0, Kind::Uaddw),
        SSUBW_Vd_Vn_Vm(i) => (i.0, Kind::Ssubw),
        SSUBW2_Vd_Vn_Vm(i) => (i.0, Kind::Ssubw),
        USUBW_Vd_Vn_Vm(i) => (i.0, Kind::Usubw),
        USUBW2_Vd_Vn_Vm(i) => (i.0, Kind::Usubw),
        SMULL_Vd_Vn_Vm(i) => (i.0, Kind::Smull),
        SMULL2_Vd_Vn_Vm(i) => (i.0, Kind::Smull),
        UMULL_Vd_Vn_Vm(i) => (i.0, Kind::Umull),
        UMULL2_Vd_Vn_Vm(i) => (i.0, Kind::Umull),
        SMLAL_Vd_Vn_Vm(i) => (i.0, Kind::Smlal),
        SMLAL2_Vd_Vn_Vm(i) => (i.0, Kind::Smlal),
        UMLAL_Vd_Vn_Vm(i) => (i.0, Kind::Umlal),
        UMLAL2_Vd_Vn_Vm(i) => (i.0, Kind::Umlal),
        SMLSL_Vd_Vn_Vm(i) => (i.0, Kind::Smlsl),
        SMLSL2_Vd_Vn_Vm(i) => (i.0, Kind::Smlsl),
        UMLSL_Vd_Vn_Vm(i) => (i.0, Kind::Umlsl),
        UMLSL2_Vd_Vn_Vm(i) => (i.0, Kind::Umlsl),
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };
    translate_with(em, raw, kind)
}

fn translate_with(em: &mut IrEmitter<'_>, raw: u32, kind: Kind) -> Result<InstStatus> {
    let high_half = bit(raw, 30) == 1;
    let size = bits(raw, 22, 2);
    let rm = bits(raw, 16, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    if size > 2 {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let vn = em.get_v_q(rn);
    let vm = em.get_v_q(rm);
    let result = match kind {
        Kind::Saddl => em.vec_saddl(vn, vm, size, high_half),
        Kind::Uaddl => em.vec_uaddl(vn, vm, size, high_half),
        Kind::Ssubl => em.vec_ssubl(vn, vm, size, high_half),
        Kind::Usubl => em.vec_usubl(vn, vm, size, high_half),
        Kind::Saddw | Kind::Uaddw | Kind::Ssubw | Kind::Usubw => {
            // Arm S/UADDW and S/USUBW keep all128 bits of wide Vn. Q selects
            // only the64-bit half of narrow Vm, then each result wraps at
            // double the source element width. Snapshot both before SetVQ.
            let z = em.const_u64(0);
            let zero = em.vec_build_q(z, z);
            let widened = if matches!(kind, Kind::Saddw | Kind::Ssubw) {
                em.vec_saddl(vm, zero, size, high_half)
            } else {
                em.vec_uaddl(vm, zero, size, high_half)
            };
            if matches!(kind, Kind::Ssubw | Kind::Usubw) {
                em.vec_sub(vn, widened, size + 1, true)
            } else {
                em.vec_add(vn, widened, size + 1, true)
            }
        }
        Kind::Smull => em.vec_smull(vn, vm, size, high_half),
        Kind::Umull => em.vec_umull(vn, vm, size, high_half),
        Kind::Smlal | Kind::Umlal | Kind::Smlsl | Kind::Umlsl => {
            // Q selects the same original 64-bit half of BOTH narrow
            // sources, never the accumulator or destination width. Capture
            // the full wide accumulator before any aliased destination write.
            let accumulator = em.get_v_q(rd);
            let product = if matches!(kind, Kind::Smlal | Kind::Smlsl) {
                em.vec_smull(vn, vm, size, high_half)
            } else {
                em.vec_umull(vn, vm, size, high_half)
            };
            if matches!(kind, Kind::Smlsl | Kind::Umlsl) {
                em.vec_sub(accumulator, product, size + 1, true)
            } else {
                em.vec_add(accumulator, product, size + 1, true)
            }
        }
    };
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
