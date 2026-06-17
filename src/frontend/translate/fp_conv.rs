use disarm64::decoder::FLOAT2INT;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: FLOAT2INT) -> Result<InstStatus> {
    use FLOAT2INT::*;
    let raw = match insn {
        FCVTZS_Rd_Fn(i)        => i.0,
        FCVTZS_Rd_W_Fn_S_D(i)  => i.0,
        SCVTF_Fd_Rn(i)         => i.0,
        SCVTF_Fd_S_D_Rn_W(i)   => i.0,
        FMOV_Fd_Rn(i)          => i.0,
        FMOV_Fd_S_S_Rn_W(i)    => i.0,
        FMOV_Rd_Fn(i)          => i.0,
        FMOV_Rd_W_Fn_S_S(i)    => i.0,
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let sf     = bit(raw, 31);
    let ptype  = bits(raw, 22, 2);
    let rmode  = bits(raw, 19, 2);
    let opcode = bits(raw, 16, 3);
    let rn     = bits(raw, 5, 5) as u8;
    let rd     = bits(raw, 0, 5) as u8;

    let dst_is_x = sf == 1;
    let src_is_x = sf == 1;
    let is_double = match ptype {
        0b00 => false,
        0b01 => true,
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }),
    };

    match (rmode, opcode) {
        (0b11, 0b000) => {
            let src = if is_double { em.get_v_d(rn) } else { em.get_v_s(rn) };
            let (op, ty) = match (is_double, dst_is_x) {
                (false, false) => (Op::FcvtZsSW, Ty::U32),
                (false, true ) => (Op::FcvtZsSX, Ty::U64),
                (true,  false) => (Op::FcvtZsDW, Ty::U32),
                (true,  true ) => (Op::FcvtZsDX, Ty::U64),
            };
            let r = em.push(Armlet::new(op, ty).with_args(&[src]));
            let size = if dst_is_x { RegSize::X } else { RegSize::W };
            em.set_gpr(rd, r, size);
        }

        (0b00, 0b010) => {
            let size = if src_is_x { RegSize::X } else { RegSize::W };
            let src = em.get_gpr(rn, size);
            let (op, ty) = match (is_double, src_is_x) {
                (false, false) => (Op::ScvtfWS, Ty::U32),
                (false, true ) => (Op::ScvtfXS, Ty::U32),
                (true,  false) => (Op::ScvtfWD, Ty::U64),
                (true,  true ) => (Op::ScvtfXD, Ty::U64),
            };
            let r = em.push(Armlet::new(op, ty).with_args(&[src]));
            if is_double { em.set_v_d(rd, r); } else { em.set_v_s(rd, r); }
        }

        (0b00, 0b111) => {
            let size = if src_is_x { RegSize::X } else { RegSize::W };
            let src = em.get_gpr(rn, size);
            if is_double { em.set_v_d(rd, src); } else { em.set_v_s(rd, src); }
        }

        (0b00, 0b110) => {
            let src = if is_double { em.get_v_d(rn) } else { em.get_v_s(rn) };
            let size = if dst_is_x { RegSize::X } else { RegSize::W };
            em.set_gpr(rd, src, size);
        }

        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }),
    }

    Ok(InstStatus::Continue)
}
