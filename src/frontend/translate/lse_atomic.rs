use disarm64::decoder::LSE_ATOMIC;

use crate::arch::{Cond, RegSize};
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty, ValueRef};
use crate::util::bits::bits;

#[derive(Clone, Copy)]
enum AtomicKind {
    Add, Clr, Eor, Set,
    Smax, Smin, Umax, Umin,
    Swp,
    Cas,
}

macro_rules! dispatch {
    (
        sz [ $( $v_sz:ident, $k_sz:ident );* $(;)? ]
        fx [ $( $v_fx:ident, $k_fx:ident, $b:expr );* $(;)? ]
    ) => {
        |em: &mut IrEmitter<'_>, insn: LSE_ATOMIC| -> Result<InstStatus> {
            match insn {
                $( LSE_ATOMIC::$v_sz(i) => atomic_rmw(em, i.0, AtomicKind::$k_sz, 0), )*
                $( LSE_ATOMIC::$v_fx(i) => atomic_rmw(em, i.0, AtomicKind::$k_fx, $b), )*
                _ => Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
            }
        }
    };
}

pub fn translate(em: &mut IrEmitter<'_>, insn: LSE_ATOMIC) -> Result<InstStatus> {
    let f = dispatch! {
        sz [
            LDADD_Rs_Rt_ADDR_SIMPLE,    Add;
            LDADDA_Rs_Rt_ADDR_SIMPLE,   Add;
            LDADDL_Rs_Rt_ADDR_SIMPLE,   Add;
            LDADDAL_Rs_Rt_ADDR_SIMPLE,  Add;
            LDCLR_Rs_Rt_ADDR_SIMPLE,    Clr;
            LDCLRA_Rs_Rt_ADDR_SIMPLE,   Clr;
            LDCLRL_Rs_Rt_ADDR_SIMPLE,   Clr;
            LDCLRAL_Rs_Rt_ADDR_SIMPLE,  Clr;
            LDEOR_Rs_Rt_ADDR_SIMPLE,    Eor;
            LDEORA_Rs_Rt_ADDR_SIMPLE,   Eor;
            LDEORL_Rs_Rt_ADDR_SIMPLE,   Eor;
            LDEORAL_Rs_Rt_ADDR_SIMPLE,  Eor;
            LDSET_Rs_Rt_ADDR_SIMPLE,    Set;
            LDSETA_Rs_Rt_ADDR_SIMPLE,   Set;
            LDSETL_Rs_Rt_ADDR_SIMPLE,   Set;
            LDSETAL_Rs_Rt_ADDR_SIMPLE,  Set;
            LDSMAX_Rs_Rt_ADDR_SIMPLE,   Smax;
            LDSMAXA_Rs_Rt_ADDR_SIMPLE,  Smax;
            LDSMAXL_Rs_Rt_ADDR_SIMPLE,  Smax;
            LDSMAXAL_Rs_Rt_ADDR_SIMPLE, Smax;
            LDSMIN_Rs_Rt_ADDR_SIMPLE,   Smin;
            LDSMINA_Rs_Rt_ADDR_SIMPLE,  Smin;
            LDSMINL_Rs_Rt_ADDR_SIMPLE,  Smin;
            LDSMINAL_Rs_Rt_ADDR_SIMPLE, Smin;
            LDUMAX_Rs_Rt_ADDR_SIMPLE,   Umax;
            LDUMAXA_Rs_Rt_ADDR_SIMPLE,  Umax;
            LDUMAXL_Rs_Rt_ADDR_SIMPLE,  Umax;
            LDUMAXAL_Rs_Rt_ADDR_SIMPLE, Umax;
            LDUMIN_Rs_Rt_ADDR_SIMPLE,   Umin;
            LDUMINA_Rs_Rt_ADDR_SIMPLE,  Umin;
            LDUMINL_Rs_Rt_ADDR_SIMPLE,  Umin;
            LDUMINAL_Rs_Rt_ADDR_SIMPLE, Umin;
            SWP_Rs_Rt_ADDR_SIMPLE,      Swp;
            SWPA_Rs_Rt_ADDR_SIMPLE,     Swp;
            SWPL_Rs_Rt_ADDR_SIMPLE,     Swp;
            SWPAL_Rs_Rt_ADDR_SIMPLE,    Swp;
            CAS_Rs_Rt_ADDR_SIMPLE,      Cas;
            CASA_Rs_Rt_ADDR_SIMPLE,     Cas;
            CASL_Rs_Rt_ADDR_SIMPLE,     Cas;
            CASAL_Rs_Rt_ADDR_SIMPLE,    Cas;
        ]
        fx [
            LDADDB_Rs_Rt_ADDR_SIMPLE,    Add,  1;
            LDADDAB_Rs_Rt_ADDR_SIMPLE,   Add,  1;
            LDADDLB_Rs_Rt_ADDR_SIMPLE,   Add,  1;
            LDADDALB_Rs_Rt_ADDR_SIMPLE,  Add,  1;
            LDADDH_Rs_Rt_ADDR_SIMPLE,    Add,  2;
            LDADDAH_Rs_Rt_ADDR_SIMPLE,   Add,  2;
            LDADDLH_Rs_Rt_ADDR_SIMPLE,   Add,  2;
            LDADDALH_Rs_Rt_ADDR_SIMPLE,  Add,  2;
            LDCLRB_Rs_Rt_ADDR_SIMPLE,    Clr,  1;
            LDCLRAB_Rs_Rt_ADDR_SIMPLE,   Clr,  1;
            LDCLRLB_Rs_Rt_ADDR_SIMPLE,   Clr,  1;
            LDCLRALB_Rs_Rt_ADDR_SIMPLE,  Clr,  1;
            LDCLRH_Rs_Rt_ADDR_SIMPLE,    Clr,  2;
            LDCLRAH_Rs_Rt_ADDR_SIMPLE,   Clr,  2;
            LDCLRLH_Rs_Rt_ADDR_SIMPLE,   Clr,  2;
            LDCLRALH_Rs_Rt_ADDR_SIMPLE,  Clr,  2;
            LDEORB_Rs_Rt_ADDR_SIMPLE,    Eor,  1;
            LDEORAB_Rs_Rt_ADDR_SIMPLE,   Eor,  1;
            LDEORLB_Rs_Rt_ADDR_SIMPLE,   Eor,  1;
            LDEORALB_Rs_Rt_ADDR_SIMPLE,  Eor,  1;
            LDEORH_Rs_Rt_ADDR_SIMPLE,    Eor,  2;
            LDEORAH_Rs_Rt_ADDR_SIMPLE,   Eor,  2;
            LDEORLH_Rs_Rt_ADDR_SIMPLE,   Eor,  2;
            LDEORALH_Rs_Rt_ADDR_SIMPLE,  Eor,  2;
            LDSETB_Rs_Rt_ADDR_SIMPLE,    Set,  1;
            LDSETAB_Rs_Rt_ADDR_SIMPLE,   Set,  1;
            LDSETLB_Rs_Rt_ADDR_SIMPLE,   Set,  1;
            LDSETALB_Rs_Rt_ADDR_SIMPLE,  Set,  1;
            LDSETH_Rs_Rt_ADDR_SIMPLE,    Set,  2;
            LDSETAH_Rs_Rt_ADDR_SIMPLE,   Set,  2;
            LDSETLH_Rs_Rt_ADDR_SIMPLE,   Set,  2;
            LDSETALH_Rs_Rt_ADDR_SIMPLE,  Set,  2;
            LDSMAXB_Rs_Rt_ADDR_SIMPLE,   Smax, 1;
            LDSMAXAB_Rs_Rt_ADDR_SIMPLE,  Smax, 1;
            LDSMAXLB_Rs_Rt_ADDR_SIMPLE,  Smax, 1;
            LDSMAXALB_Rs_Rt_ADDR_SIMPLE, Smax, 1;
            LDSMAXH_Rs_Rt_ADDR_SIMPLE,   Smax, 2;
            LDSMAXAH_Rs_Rt_ADDR_SIMPLE,  Smax, 2;
            LDSMAXLH_Rs_Rt_ADDR_SIMPLE,  Smax, 2;
            LDSMAXALH_Rs_Rt_ADDR_SIMPLE, Smax, 2;
            LDSMINB_Rs_Rt_ADDR_SIMPLE,   Smin, 1;
            LDSMINAB_Rs_Rt_ADDR_SIMPLE,  Smin, 1;
            LDSMINLB_Rs_Rt_ADDR_SIMPLE,  Smin, 1;
            LDSMINALB_Rs_Rt_ADDR_SIMPLE, Smin, 1;
            LDSMINH_Rs_Rt_ADDR_SIMPLE,   Smin, 2;
            LDSMINAH_Rs_Rt_ADDR_SIMPLE,  Smin, 2;
            LDSMINLH_Rs_Rt_ADDR_SIMPLE,  Smin, 2;
            LDSMINALH_Rs_Rt_ADDR_SIMPLE, Smin, 2;
            LDUMAXB_Rs_Rt_ADDR_SIMPLE,   Umax, 1;
            LDUMAXAB_Rs_Rt_ADDR_SIMPLE,  Umax, 1;
            LDUMAXLB_Rs_Rt_ADDR_SIMPLE,  Umax, 1;
            LDUMAXALB_Rs_Rt_ADDR_SIMPLE, Umax, 1;
            LDUMAXH_Rs_Rt_ADDR_SIMPLE,   Umax, 2;
            LDUMAXAH_Rs_Rt_ADDR_SIMPLE,  Umax, 2;
            LDUMAXLH_Rs_Rt_ADDR_SIMPLE,  Umax, 2;
            LDUMAXALH_Rs_Rt_ADDR_SIMPLE, Umax, 2;
            LDUMINB_Rs_Rt_ADDR_SIMPLE,   Umin, 1;
            LDUMINAB_Rs_Rt_ADDR_SIMPLE,  Umin, 1;
            LDUMINLB_Rs_Rt_ADDR_SIMPLE,  Umin, 1;
            LDUMINALB_Rs_Rt_ADDR_SIMPLE, Umin, 1;
            LDUMINH_Rs_Rt_ADDR_SIMPLE,   Umin, 2;
            LDUMINAH_Rs_Rt_ADDR_SIMPLE,  Umin, 2;
            LDUMINLH_Rs_Rt_ADDR_SIMPLE,  Umin, 2;
            LDUMINALH_Rs_Rt_ADDR_SIMPLE, Umin, 2;
            SWPB_Rs_Rt_ADDR_SIMPLE,      Swp,  1;
            SWPAB_Rs_Rt_ADDR_SIMPLE,     Swp,  1;
            SWPLB_Rs_Rt_ADDR_SIMPLE,     Swp,  1;
            SWPALB_Rs_Rt_ADDR_SIMPLE,    Swp,  1;
            SWPH_Rs_Rt_ADDR_SIMPLE,      Swp,  2;
            SWPAH_Rs_Rt_ADDR_SIMPLE,     Swp,  2;
            SWPLH_Rs_Rt_ADDR_SIMPLE,     Swp,  2;
            SWPALH_Rs_Rt_ADDR_SIMPLE,    Swp,  2;
            CASB_Rs_Rt_ADDR_SIMPLE,      Cas,  1;
            CASAB_Rs_Rt_ADDR_SIMPLE,     Cas,  1;
            CASLB_Rs_Rt_ADDR_SIMPLE,     Cas,  1;
            CASALB_Rs_Rt_ADDR_SIMPLE,    Cas,  1;
            CASH_Rs_Rt_ADDR_SIMPLE,      Cas,  2;
            CASAH_Rs_Rt_ADDR_SIMPLE,     Cas,  2;
            CASLH_Rs_Rt_ADDR_SIMPLE,     Cas,  2;
            CASALH_Rs_Rt_ADDR_SIMPLE,    Cas,  2;
        ]
    };
    f(em, insn)
}

fn atomic_rmw(
    em: &mut IrEmitter<'_>,
    raw: u32,
    kind: AtomicKind,
    fixed_bytes: u32,
) -> Result<InstStatus> {
    let bytes = if fixed_bytes != 0 {
        fixed_bytes
    } else {
        1u32 << bits(raw, 30, 2)
    };
    let rs = bits(raw, 16, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rt = bits(raw, 0, 5) as u8;
    let size = if bytes == 8 { RegSize::X } else { RegSize::W };

    let addr = em.get_x_or_sp(rn, true);
    let rs_val = em.get_gpr(rs, size);

    let old = em.load(addr, bytes);

    match kind {
        AtomicKind::Cas => {
            let rt_val = em.get_gpr(rt, size);
            em.subs(old, rs_val, size);
            let new_val = csel(em, rt_val, old, Cond::EQ, size);
            em.store(addr, new_val, bytes);
            em.set_gpr(rs, old, size);
        }
        AtomicKind::Swp => {
            em.store(addr, rs_val, bytes);
            em.set_gpr(rt, old, size);
        }
        _ => {
            let new_val = apply_op(em, old, rs_val, kind, size);
            em.store(addr, new_val, bytes);
            em.set_gpr(rt, old, size);
        }
    }

    Ok(InstStatus::Continue)
}

fn apply_op(
    em: &mut IrEmitter<'_>,
    old: ValueRef,
    rs: ValueRef,
    kind: AtomicKind,
    size: RegSize,
) -> ValueRef {
    match kind {
        AtomicKind::Add => em.add(old, rs, size),
        AtomicKind::Clr => {
            let not_rs = bitwise_not(em, rs, size);
            em.and(old, not_rs, size)
        }
        AtomicKind::Eor => em.eor(old, rs, size),
        AtomicKind::Set => em.or(old, rs, size),
        AtomicKind::Smax => signed_select(em, old, rs, Cond::GT, size),
        AtomicKind::Smin => signed_select(em, old, rs, Cond::LT, size),
        AtomicKind::Umax => signed_select(em, old, rs, Cond::HI, size),
        AtomicKind::Umin => signed_select(em, old, rs, Cond::CC, size),
        AtomicKind::Swp | AtomicKind::Cas => unreachable!(),
    }
}

fn bitwise_not(em: &mut IrEmitter<'_>, v: ValueRef, size: RegSize) -> ValueRef {
    let (op, ty) = match size {
        RegSize::W => (Op::Not32, Ty::U32),
        RegSize::X => (Op::Not64, Ty::U64),
    };
    em.push(Armlet::new(op, ty).with_args(&[v]))
}

fn signed_select(
    em: &mut IrEmitter<'_>,
    old: ValueRef,
    rs: ValueRef,
    cond: Cond,
    size: RegSize,
) -> ValueRef {
    em.subs(old, rs, size);
    csel(em, old, rs, cond, size)
}

fn csel(em: &mut IrEmitter<'_>, a: ValueRef, b: ValueRef, cond: Cond, size: RegSize) -> ValueRef {
    let (op, ty) = match size {
        RegSize::W => (Op::Csel32, Ty::U32),
        RegSize::X => (Op::Csel64, Ty::U64),
    };
    let nzcv = em.get_nzcv();
    em.push(Armlet::new(op, ty)
        .with_args(&[a, b, nzcv])
        .with_imm(cond as u64))
}
