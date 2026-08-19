use disarm64::decoder::LDSTEXCL;

use crate::arch::{RegSize, ZR_ENCODING};
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::bits;

enum Kind {
    LoadEx { bytes: u32, target_x: bool },
    StoreEx { bytes: u32 },
    Load { bytes: u32, target_x: bool },
    Store { bytes: u32 },
}

fn size_from_bit30(raw: u32) -> (u32, bool) {
    let size = bits(raw, 30, 2);
    let bytes = 1u32 << size;
    (bytes, bytes == 8)
}

pub fn translate(em: &mut IrEmitter<'_>, insn: LDSTEXCL) -> Result<InstStatus> {
    use LDSTEXCL::*;
    let (raw, kind) = match insn {
        LDXRB_Rt_ADDR_SIMPLE(i) => (
            i.0,
            Kind::LoadEx {
                bytes: 1,
                target_x: false,
            },
        ),
        LDAXRB_Rt_ADDR_SIMPLE(i) => (
            i.0,
            Kind::LoadEx {
                bytes: 1,
                target_x: false,
            },
        ),
        LDXRH_Rt_ADDR_SIMPLE(i) => (
            i.0,
            Kind::LoadEx {
                bytes: 2,
                target_x: false,
            },
        ),
        LDAXRH_Rt_ADDR_SIMPLE(i) => (
            i.0,
            Kind::LoadEx {
                bytes: 2,
                target_x: false,
            },
        ),
        LDXR_Rt_ADDR_SIMPLE(i) => {
            let (b, tx) = size_from_bit30(i.0);
            (
                i.0,
                Kind::LoadEx {
                    bytes: b,
                    target_x: tx,
                },
            )
        }
        LDAXR_Rt_ADDR_SIMPLE(i) => {
            let (b, tx) = size_from_bit30(i.0);
            (
                i.0,
                Kind::LoadEx {
                    bytes: b,
                    target_x: tx,
                },
            )
        }

        STXRB_Rs_Rt_ADDR_SIMPLE(i) => (i.0, Kind::StoreEx { bytes: 1 }),
        STLXRB_Rs_Rt_ADDR_SIMPLE(i) => (i.0, Kind::StoreEx { bytes: 1 }),
        STXRH_Rs_Rt_ADDR_SIMPLE(i) => (i.0, Kind::StoreEx { bytes: 2 }),
        STLXRH_Rs_Rt_ADDR_SIMPLE(i) => (i.0, Kind::StoreEx { bytes: 2 }),
        STXR_Rs_Rt_ADDR_SIMPLE(i) => {
            let (b, _) = size_from_bit30(i.0);
            (i.0, Kind::StoreEx { bytes: b })
        }
        STLXR_Rs_Rt_ADDR_SIMPLE(i) => {
            let (b, _) = size_from_bit30(i.0);
            (i.0, Kind::StoreEx { bytes: b })
        }

        LDARB_Rt_ADDR_SIMPLE(i) => (
            i.0,
            Kind::Load {
                bytes: 1,
                target_x: false,
            },
        ),
        LDAPRB_Rt_ADDR_SIMPLE(i) => (
            i.0,
            Kind::Load {
                bytes: 1,
                target_x: false,
            },
        ),
        LDLARB_Rt_ADDR_SIMPLE(i) => (
            i.0,
            Kind::Load {
                bytes: 1,
                target_x: false,
            },
        ),
        LDARH_Rt_ADDR_SIMPLE(i) => (
            i.0,
            Kind::Load {
                bytes: 2,
                target_x: false,
            },
        ),
        LDAPRH_Rt_ADDR_SIMPLE(i) => (
            i.0,
            Kind::Load {
                bytes: 2,
                target_x: false,
            },
        ),
        LDLARH_Rt_ADDR_SIMPLE(i) => (
            i.0,
            Kind::Load {
                bytes: 2,
                target_x: false,
            },
        ),
        LDAR_Rt_ADDR_SIMPLE(i) => {
            let (b, tx) = size_from_bit30(i.0);
            (
                i.0,
                Kind::Load {
                    bytes: b,
                    target_x: tx,
                },
            )
        }
        LDAPR_Rt_ADDR_SIMPLE(i) => {
            let (b, tx) = size_from_bit30(i.0);
            (
                i.0,
                Kind::Load {
                    bytes: b,
                    target_x: tx,
                },
            )
        }
        LDLAR_Rt_ADDR_SIMPLE(i) => {
            let (b, tx) = size_from_bit30(i.0);
            (
                i.0,
                Kind::Load {
                    bytes: b,
                    target_x: tx,
                },
            )
        }

        STLRB_Rt_ADDR_SIMPLE(i) => (i.0, Kind::Store { bytes: 1 }),
        STLLRB_Rt_ADDR_SIMPLE(i) => (i.0, Kind::Store { bytes: 1 }),
        STLRH_Rt_ADDR_SIMPLE(i) => (i.0, Kind::Store { bytes: 2 }),
        STLLRH_Rt_ADDR_SIMPLE(i) => (i.0, Kind::Store { bytes: 2 }),
        STLR_Rt_ADDR_SIMPLE(i) => {
            let (b, _) = size_from_bit30(i.0);
            (i.0, Kind::Store { bytes: b })
        }
        STLLR_Rt_ADDR_SIMPLE(i) => {
            let (b, _) = size_from_bit30(i.0);
            (i.0, Kind::Store { bytes: b })
        }

        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };

    let rn = bits(raw, 5, 5) as u8;
    let rt = bits(raw, 0, 5) as u8;
    let addr = em.get_x_or_sp(rn, true);

    match kind {
        Kind::LoadEx { bytes, target_x } => {
            let (op, ty) = match bytes {
                1 => (Op::LoadEx8, Ty::U32),
                2 => (Op::LoadEx16, Ty::U32),
                4 => (Op::LoadEx32, Ty::U32),
                8 => (Op::LoadEx64, Ty::U64),
                _ => unreachable!(),
            };
            let value = em.push(Armlet::new(op, ty).with_args(&[addr]));
            if target_x {
                em.set_x(rt, value);
            } else {
                em.set_w(rt, value);
            }
        }
        Kind::StoreEx { bytes } => {
            let rs = bits(raw, 16, 5) as u8;
            let val_size = if bytes == 8 { RegSize::X } else { RegSize::W };
            let val = em.get_gpr(rt, val_size);
            let op = match bytes {
                1 => Op::StoreEx8,
                2 => Op::StoreEx16,
                4 => Op::StoreEx32,
                8 => Op::StoreEx64,
                _ => unreachable!(),
            };
            let result = em.push(Armlet::new(op, Ty::U32).with_args(&[addr, val]));
            if rs != ZR_ENCODING {
                em.set_w(rs, result);
            }
        }
        Kind::Load { bytes, target_x } => {
            let v = em.load(addr, bytes);
            if target_x {
                em.set_x(rt, v);
            } else {
                em.set_w(rt, v);
            }
        }
        Kind::Store { bytes } => {
            let val_size = if bytes == 8 { RegSize::X } else { RegSize::W };
            let val = em.get_gpr(rt, val_size);
            em.store(addr, val, bytes);
        }
    }
    Ok(InstStatus::Continue)
}
