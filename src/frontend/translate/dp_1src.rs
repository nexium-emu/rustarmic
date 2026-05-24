use disarm64::decoder::DP_1SRC;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum Kind { Clz, Cls, Rbit, Rev16, Rev32, RevAll }

pub fn translate(em: &mut IrEmitter<'_>, insn: DP_1SRC) -> Result<InstStatus> {
    use DP_1SRC::*;
    let (raw, kind) = match insn {
        CLZ_Rd_Rn(i)      => (i.0, Kind::Clz),
        CLS_Rd_Rn(i)      => (i.0, Kind::Cls),
        RBIT_Rd_Rn(i)     => (i.0, Kind::Rbit),
        REV16_Rd_Rn(i)    => (i.0, Kind::Rev16),
        REV32_Rd_Rn(i)    => (i.0, Kind::Rev32),
        REV_Rd_Rn(i)      => (i.0, Kind::RevAll),
        REV_Rd_X_Rn_X(i)  => (i.0, Kind::RevAll),

        PACIA_Rd_Rn_SP(i)  => return pac_identity(em, i.0),
        PACIB_Rd_Rn_SP(i)  => return pac_identity(em, i.0),
        PACDA_Rd_Rn_SP(i)  => return pac_identity(em, i.0),
        PACDB_Rd_Rn_SP(i)  => return pac_identity(em, i.0),
        AUTIA_Rd_Rn_SP(i)  => return pac_identity(em, i.0),
        AUTIB_Rd_Rn_SP(i)  => return pac_identity(em, i.0),
        AUTDA_Rd_Rn_SP(i)  => return pac_identity(em, i.0),
        AUTDB_Rd_Rn_SP(i)  => return pac_identity(em, i.0),

        PACIZA_Rd(i) => return pac_identity(em, i.0),
        PACIZB_Rd(i) => return pac_identity(em, i.0),
        PACDZA_Rd(i) => return pac_identity(em, i.0),
        PACDZB_Rd(i) => return pac_identity(em, i.0),
        AUTIZA_Rd(i) => return pac_identity(em, i.0),
        AUTIZB_Rd(i) => return pac_identity(em, i.0),
        AUTDZA_Rd(i) => return pac_identity(em, i.0),
        AUTDZB_Rd(i) => return pac_identity(em, i.0),
        XPACI_Rd(i)  => return pac_identity(em, i.0),
        XPACD_Rd(i)  => return pac_identity(em, i.0),
    };

    let sf = bit(raw, 31);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let ty   = if sf == 1 { Ty::U64 } else { Ty::U32 };

    let src = em.get_gpr(rn, size);
    let (op, op_ty) = match (kind, sf) {
        (Kind::Clz,    0) => (Op::Clz32, ty),
        (Kind::Clz,    1) => (Op::Clz64, ty),
        (Kind::Cls,    0) => (Op::Cls32, ty),
        (Kind::Cls,    1) => (Op::Cls64, ty),
        (Kind::Rbit,   0) => (Op::Rbit32, ty),
        (Kind::Rbit,   1) => (Op::Rbit64, ty),
        (Kind::Rev16,  _) => (Op::Rev16, ty),
        (Kind::Rev32,  1) => (Op::Rev32, ty),
        (Kind::RevAll, _) => (Op::Rev64, ty),
        _ => return Err(Error::Decode { pc: em.current_pc, opcode: raw }),
    };
    let result = em.push(Armlet::new(op, op_ty).with_args(&[src]));
    em.set_gpr(rd, result, size);
    Ok(InstStatus::Continue)
}

fn pac_identity(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let rd = bits(raw, 0, 5) as u8;
    let src = em.get_gpr(rd, RegSize::X);
    let pass = em.push(Armlet::new(Op::Identity, Ty::U64).with_args(&[src]));
    em.set_gpr(rd, pass, RegSize::X);
    Ok(InstStatus::Continue)
}
