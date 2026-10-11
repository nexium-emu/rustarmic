//! Lowers an instruction to one out-of-line helper call (see `interp`).

use crate::error::Result;
use crate::frontend::translator::InstStatus;
use crate::interp::{Dst, Plan, Src};
use crate::ir::{IrEmitter, Ty, ValueRef};
use crate::jit::helpers;

pub fn translate(em: &mut IrEmitter<'_>, inst: u32, plan: Plan) -> Result<InstStatus> {
    let mut args = [ValueRef::NONE; 4];
    let mut count = 0;
    for (slot, src) in plan.srcs.iter().enumerate() {
        let Some(src) = *src else {
            continue;
        };
        // Operand slots are positional; skipped slots keep stale data the
        // helper does not read, but a later present slot must not shift.
        while count < slot {
            args[count] = em.const_u32(0);
            count += 1;
        }
        args[count] = match src {
            Src::V(r) => em.get_v_q(r),
            Src::X(r) => em.get_x(r),
            Src::W(r) => em.get_w(r),
        };
        count += 1;
    }
    let ty = match plan.dst {
        Dst::V(_) => Ty::U128,
        Dst::X(_) => Ty::U64,
        Dst::W(_) => Ty::U32,
    };
    let r = em.call_helper(helpers::INTERP, inst, &args[..count], ty);
    match plan.dst {
        Dst::V(rd) => em.set_v_q(rd, r),
        Dst::X(rd) => em.set_x(rd, r),
        Dst::W(rd) => em.set_w(rd, r),
    }
    Ok(InstStatus::Continue)
}
