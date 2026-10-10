use disarm64::decoder::DP_2SRC;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

enum Kind {
    Lslv,
    Lsrv,
    Asrv,
    Rorv,
    Udiv,
    Sdiv,
}

pub fn translate(em: &mut IrEmitter<'_>, insn: DP_2SRC) -> Result<InstStatus> {
    use DP_2SRC::*;
    let (raw, kind) = match insn {
        LSLV_Rd_Rn_Rm(i) => (i.0, Kind::Lslv),
        LSRV_Rd_Rn_Rm(i) => (i.0, Kind::Lsrv),
        ASRV_Rd_Rn_Rm(i) => (i.0, Kind::Asrv),
        RORV_Rd_Rn_Rm(i) => (i.0, Kind::Rorv),
        UDIV_Rd_Rn_Rm(i) => (i.0, Kind::Udiv),
        SDIV_Rd_Rn_Rm(i) => (i.0, Kind::Sdiv),
        CRC32B_Rd_Rn_Rm(i) => return translate_crc(em, i.0),
        CRC32H_Rd_Rn_Rm(i) => return translate_crc(em, i.0),
        CRC32W_Rd_Rn_Rm(i) => return translate_crc(em, i.0),
        CRC32X_Rd_Rn_Rm(i) => return translate_crc(em, i.0),
        CRC32CB_Rd_Rn_Rm(i) => return translate_crc(em, i.0),
        CRC32CH_Rd_Rn_Rm(i) => return translate_crc(em, i.0),
        CRC32CW_Rd_Rn_Rm(i) => return translate_crc(em, i.0),
        CRC32CX_Rd_Rn_Rm(i) => return translate_crc(em, i.0),
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };

    let sf = bit(raw, 31);
    let rm = bits(raw, 16, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let ty = if sf == 1 { Ty::U64 } else { Ty::U32 };

    let n = em.get_gpr(rn, size);
    let m = em.get_gpr(rm, size);

    let result = match kind {
        Kind::Lslv => em.lsl(n, m, size),
        Kind::Lsrv => em.lsr(n, m, size),
        Kind::Asrv => em.asr(n, m, size),
        Kind::Rorv => em.ror(n, m, size),
        Kind::Udiv => {
            let op = if sf == 1 { Op::UDiv64 } else { Op::UDiv32 };
            em.push(Armlet::new(op, ty).with_args(&[n, m]))
        }
        Kind::Sdiv => {
            let op = if sf == 1 { Op::SDiv64 } else { Op::SDiv32 };
            em.push(Armlet::new(op, ty).with_args(&[n, m]))
        }
    };
    em.set_gpr(rd, result, size);
    Ok(InstStatus::Continue)
}

fn translate_crc(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let width = bits(raw, 10, 2);
    let op = match (bit(raw, 12), width) {
        (0, 0) => Op::Crc32Ieee8,
        (0, 1) => Op::Crc32Ieee16,
        (0, 2) => Op::Crc32Ieee32,
        (0, 3) => Op::Crc32Ieee64,
        (1, 0) => Op::Crc32Castagnoli8,
        (1, 1) => Op::Crc32Castagnoli16,
        (1, 2) => Op::Crc32Castagnoli32,
        (1, 3) => Op::Crc32Castagnoli64,
        _ => unreachable!(),
    };
    let rn = bits(raw, 5, 5) as u8;
    let rm = bits(raw, 16, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let accumulator = em.get_w(rn);
    let value = em.get_gpr(rm, if width == 3 { RegSize::X } else { RegSize::W });
    let result = em.push(Armlet::new(op, Ty::U32).with_args(&[accumulator, value]));
    // Even CRC32X has a W accumulator and a zero-extending W destination.
    em.set_w(rd, result);
    Ok(InstStatus::Continue)
}
