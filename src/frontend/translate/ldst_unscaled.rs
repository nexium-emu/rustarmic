use disarm64::decoder::LDST_UNSCALED;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bits, sign_extend};

enum Kind { LoadU, LoadS, Store }

pub fn translate(em: &mut IrEmitter<'_>, insn: LDST_UNSCALED) -> Result<InstStatus> {
    use LDST_UNSCALED::*;
    let (raw, kind, target_x) = match insn {
        LDUR_Rt_ADDR_SIMM9(i)   => (i.0, Kind::LoadU, true),
        LDURB_Rt_ADDR_SIMM9(i)  => (i.0, Kind::LoadU, false),
        LDURH_Rt_ADDR_SIMM9(i)  => (i.0, Kind::LoadU, false),
        LDURSB_Rt_ADDR_SIMM9(i) => (i.0, Kind::LoadS, true),
        LDURSH_Rt_ADDR_SIMM9(i) => (i.0, Kind::LoadS, true),
        LDURSW_Rt_ADDR_SIMM9(i) => (i.0, Kind::LoadS, true),
        STUR_Rt_ADDR_SIMM9(i)   => (i.0, Kind::Store, true),
        STURB_Rt_ADDR_SIMM9(i)  => (i.0, Kind::Store, false),
        STURH_Rt_ADDR_SIMM9(i)  => (i.0, Kind::Store, false),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let size  = bits(raw, 30, 2);
    let imm9  = bits(raw, 12, 9);
    let rn    = bits(raw, 5, 5) as u8;
    let rt    = bits(raw, 0, 5) as u8;

    let bytes  = 1u32 << size;
    let offset = sign_extend(imm9 as u64, 9);

    let base = em.get_x_or_sp(rn, true);
    let off  = em.const_u64(offset as u64);
    let addr = em.add(base, off, RegSize::X);

    match kind {
        Kind::Store => {
            let val_size = if size == 3 { RegSize::X } else { RegSize::W };
            let val = em.get_gpr(rt, val_size);
            em.store(addr, val, bytes);
        }
        Kind::LoadU => {
            let v = em.load(addr, bytes);
            if size <= 2 { em.set_w(rt, v); } else { em.set_x(rt, v); }
            let _ = target_x;
        }
        Kind::LoadS => {
            let v = em.load(addr, bytes);
            let width_bits = (bytes * 8) as u64;
            let shl = em.const_u64(64 - width_bits);
            let s1 = em.lsl(v, shl, RegSize::X);
            let shl2 = em.const_u64(64 - width_bits);
            let sx = em.asr(s1, shl2, RegSize::X);
            if target_x { em.set_x(rt, sx); } else { em.set_w(rt, sx); }
        }
    }
    Ok(InstStatus::Continue)
}
