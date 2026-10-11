use disarm64::decoder::LDST_UNSCALED;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits, sign_extend};

enum Kind {
    LoadU,
    LoadS,
    Store,
    FpLoad,
    FpStore,
}

pub fn translate(em: &mut IrEmitter<'_>, insn: LDST_UNSCALED) -> Result<InstStatus> {
    use LDST_UNSCALED::*;
    let (raw, kind, target_x) = match insn {
        // Prefetch is an implementation-defined hint, not a load. This JIT
        // deliberately does not prefetch; inaccessible addresses must not
        // become ordinary Memory callbacks or guest data aborts.
        PRFUM_PRFOP_ADDR_SIMM9(_) => return Ok(InstStatus::Continue),
        LDUR_Rt_ADDR_SIMM9(i) => (i.0, Kind::LoadU, true),
        LDURB_Rt_ADDR_SIMM9(i) => (i.0, Kind::LoadU, false),
        LDURH_Rt_ADDR_SIMM9(i) => (i.0, Kind::LoadU, false),
        // Signed loads: opc=10 targets X; opc=11 targets W (opc<0>=bit22).
        LDURSB_Rt_ADDR_SIMM9(i) => (i.0, Kind::LoadS, bit(i.0, 22) == 0),
        LDURSH_Rt_ADDR_SIMM9(i) => (i.0, Kind::LoadS, bit(i.0, 22) == 0),
        LDURSW_Rt_ADDR_SIMM9(i) => (i.0, Kind::LoadS, true),
        STUR_Rt_ADDR_SIMM9(i) => (i.0, Kind::Store, true),
        STURB_Rt_ADDR_SIMM9(i) => (i.0, Kind::Store, false),
        STURH_Rt_ADDR_SIMM9(i) => (i.0, Kind::Store, false),
        LDUR_Ft_ADDR_SIMM9(i) => (i.0, Kind::FpLoad, false),
        STUR_Ft_ADDR_SIMM9(i) => (i.0, Kind::FpStore, false),
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };

    let size = bits(raw, 30, 2);
    let imm9 = bits(raw, 12, 9);
    let rn = bits(raw, 5, 5) as u8;
    let rt = bits(raw, 0, 5) as u8;

    let q_form = matches!(kind, Kind::FpLoad | Kind::FpStore) && size == 0 && bit(raw, 23) == 1;
    let bytes = if q_form { 16 } else { 1u32 << size };
    let offset = sign_extend(imm9 as u64, 9);

    let base = em.get_x_or_sp(rn, true);
    let off = em.const_u64(offset as u64);
    let addr = em.add(base, off, RegSize::X);

    match kind {
        Kind::Store => {
            let val_size = if size == 3 { RegSize::X } else { RegSize::W };
            let val = em.get_gpr(rt, val_size);
            em.store(addr, val, bytes);
        }
        Kind::LoadU => {
            let v = em.load(addr, bytes);
            if size <= 2 {
                em.set_w(rt, v);
            } else {
                em.set_x(rt, v);
            }
            let _ = target_x;
        }
        Kind::LoadS => {
            let v = em.load(addr, bytes);
            let width_bits = (bytes * 8) as u64;
            let shl = em.const_u64(64 - width_bits);
            let s1 = em.lsl(v, shl, RegSize::X);
            let shl2 = em.const_u64(64 - width_bits);
            let sx = em.asr(s1, shl2, RegSize::X);
            if target_x {
                em.set_x(rt, sx);
            } else {
                em.set_w(rt, sx);
            }
        }
        Kind::FpLoad => {
            if bytes == 16 {
                let q = em.load(addr, 16);
                em.set_v_q(rt, q);
            } else if bytes == 8 {
                let v = em.load(addr, bytes);
                em.set_v_d(rt, v);
            } else if matches!(bytes, 1 | 2 | 4) {
                let v = em.load(addr, bytes);
                // SetV accepts a U32 scalar, not Load8/16's narrow IR type.
                // Promote with an explicit mask and clear the entire upper Q.
                let v = if bytes < 4 {
                    let mask = em.const_u32((1u32 << (bytes * 8)) - 1);
                    em.and(v, mask, RegSize::W)
                } else {
                    v
                };
                em.set_v_s(rt, v);
            } else {
                return Err(Error::Unsupported {
                    pc: em.current_pc,
                    opcode: raw,
                });
            }
        }
        Kind::FpStore => {
            if bytes == 16 {
                let q = em.get_v_q(rt);
                em.store(addr, q, 16);
            } else {
                let v = if bytes == 8 {
                    em.get_v_d(rt)
                } else if matches!(bytes, 1 | 2 | 4) {
                    em.get_v_s(rt)
                } else {
                    return Err(Error::Unsupported {
                        pc: em.current_pc,
                        opcode: raw,
                    });
                };
                em.store(addr, v, bytes);
            }
        }
    }
    Ok(InstStatus::Continue)
}
