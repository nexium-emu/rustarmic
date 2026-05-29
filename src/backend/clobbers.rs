use bitflags::bitflags;

use crate::ir::Op;

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
    pub struct GprMask: u16 {
        const RAX = 1 << 0;
        const RCX = 1 << 1;
        const RDX = 1 << 2;
        const RBX = 1 << 3;
        const RSP = 1 << 4;
        const RBP = 1 << 5;
        const RSI = 1 << 6;
        const RDI = 1 << 7;
        const R8  = 1 << 8;
        const R9  = 1 << 9;
        const R10 = 1 << 10;
        const R11 = 1 << 11;
        const R12 = 1 << 12;
        const R13 = 1 << 13;
        const R14 = 1 << 14;
        const R15 = 1 << 15;
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
    pub struct XmmMask: u16 {
        const XMM0  = 1 << 0;
        const XMM1  = 1 << 1;
        const XMM2  = 1 << 2;
        const XMM3  = 1 << 3;
        const XMM4  = 1 << 4;
        const XMM5  = 1 << 5;
        const XMM6  = 1 << 6;
        const XMM7  = 1 << 7;
        const XMM8  = 1 << 8;
        const XMM9  = 1 << 9;
        const XMM10 = 1 << 10;
        const XMM11 = 1 << 11;
        const XMM12 = 1 << 12;
        const XMM13 = 1 << 13;
        const XMM14 = 1 << 14;
        const XMM15 = 1 << 15;
    }
}

pub const CALLER_SAVED_GPRS: GprMask = GprMask::from_bits_retain(
    GprMask::RAX.bits() | GprMask::RCX.bits() | GprMask::RDX.bits()
    | GprMask::RSI.bits() | GprMask::RDI.bits()
    | GprMask::R8.bits() | GprMask::R9.bits()
    | GprMask::R10.bits() | GprMask::R11.bits(),
);

/// XMM registers a host `extern "C"` callee may freely trash. Windows x64
/// preserves XMM6..XMM15 (callee-saved) so only XMM0..XMM5 are caller-saved;
/// SysV AMD64 treats every XMM as caller-saved. The regalloc uses this to
/// keep U128 values whose live range crosses a hook-emitting op (Load*,
/// Store*, Mrs CNTPCT) out of XMM slots that the callback would corrupt.
#[cfg(target_os = "windows")]
pub const CALLER_SAVED_XMMS: XmmMask = XmmMask::from_bits_retain(
    XmmMask::XMM0.bits() | XmmMask::XMM1.bits() | XmmMask::XMM2.bits()
    | XmmMask::XMM3.bits() | XmmMask::XMM4.bits() | XmmMask::XMM5.bits(),
);
#[cfg(not(target_os = "windows"))]
pub const CALLER_SAVED_XMMS: XmmMask = XmmMask::from_bits_retain(0xFFFF);

pub mod gpr_id {
    pub const RAX: u8 = 0;
    pub const RCX: u8 = 1;
    pub const RDX: u8 = 2;
    pub const RBX: u8 = 3;
    pub const RSI: u8 = 6;
    pub const RDI: u8 = 7;
    pub const R8:  u8 = 8;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ClobberSet {
    pub gpr: GprMask,
    pub xmm: XmmMask,
    pub result_pinned_to_gpr: Option<u8>,
    pub result_pinned_to_xmm: Option<u8>,
}

pub fn clobbers_for_op(op: Op) -> ClobberSet {
    use Op::*;
    let mut c = ClobberSet::default();

    match op {
        UDiv32 | UDiv64 | SDiv32 | SDiv64 => {
            c.gpr = GprMask::RAX | GprMask::RCX | GprMask::RDX;
            c.result_pinned_to_gpr = Some(gpr_id::RAX);
        }

        Lsl32 | Lsl64 | Lsr32 | Lsr64
        | Asr32 | Asr64 | Ror32 | Ror64 => {
            c.gpr = GprMask::RAX | GprMask::RCX;
            c.result_pinned_to_gpr = Some(gpr_id::RAX);
        }

        AddsFlags32 | AddsFlags64 | SubsFlags32 | SubsFlags64 => {
            c.gpr = GprMask::RAX | GprMask::RCX | GprMask::RSI
                  | GprMask::R8  | GprMask::R9
                  | GprMask::R10 | GprMask::R11;
            c.result_pinned_to_gpr = Some(gpr_id::RAX);
        }

        Adc32 | Adc64 | Sbc32 | Sbc64 => {
            c.gpr = GprMask::RAX | GprMask::RCX;
            c.result_pinned_to_gpr = Some(gpr_id::RAX);
        }

        Clz32 | Clz64 | Cls32 | Cls64 => {
            c.gpr = GprMask::RAX | GprMask::RCX;
            c.result_pinned_to_gpr = Some(gpr_id::RAX);
        }

        Rbit32 | Rbit64 | Rev16 | Rev32 | Rev64 => {
            c.gpr = GprMask::RAX | GprMask::RCX | GprMask::RDX;
            c.result_pinned_to_gpr = Some(gpr_id::RAX);
        }

        Csel32 | Csel64 => {
            c.gpr = GprMask::RAX | GprMask::RCX | GprMask::RDX
                  | GprMask::RSI | GprMask::RDI | GprMask::R8 | GprMask::R9;
            c.result_pinned_to_gpr = Some(gpr_id::RAX);
        }

        Load8 | Load16 | Load32 | Load64 | Load128
        | Store8 | Store16 | Store32 | Store64 | Store128
        | LoadEx8 | LoadEx16 | LoadEx32 | LoadEx64
        | StoreEx8 | StoreEx16 | StoreEx32 | StoreEx64 => {
            c.gpr = CALLER_SAVED_GPRS;
            c.xmm = CALLER_SAVED_XMMS;
            if matches!(op, Load8 | Load16 | Load32 | Load64
                | LoadEx8 | LoadEx16 | LoadEx32 | LoadEx64
                | StoreEx8 | StoreEx16 | StoreEx32 | StoreEx64)
            {
                c.result_pinned_to_gpr = Some(gpr_id::RAX);
            }
        }

        Mrs => {
            // CNTPCT_EL0 / CNTVCT_EL0 fall through to an indirect call to
            // read_cntpct; other sysregs are inline ctx-relative mov. The
            // imm is hidden from us here, so be conservative and mark every
            // Mrs as a clobber barrier.
            c.gpr = CALLER_SAVED_GPRS;
            c.xmm = CALLER_SAVED_XMMS;
            c.result_pinned_to_gpr = Some(gpr_id::RAX);
        }

        _ => {
            c.gpr = GprMask::RAX | GprMask::RCX | GprMask::RDX
                  | GprMask::RSI | GprMask::RDI | GprMask::R8;
            c.result_pinned_to_gpr = Some(gpr_id::RAX);
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn div_clobbers_rax_rcx_rdx_and_pins_rax() {
        let c = clobbers_for_op(Op::SDiv64);
        assert!(c.gpr.contains(GprMask::RAX));
        assert!(c.gpr.contains(GprMask::RCX));
        assert!(c.gpr.contains(GprMask::RDX));
        assert_eq!(c.result_pinned_to_gpr, Some(gpr_id::RAX));
    }

    #[test]
    fn shifts_need_rcx_for_cl() {
        for op in [Op::Lsl64, Op::Lsr64, Op::Asr64, Op::Ror64] {
            let c = clobbers_for_op(op);
            assert!(c.gpr.contains(GprMask::RCX),
                "{:?} should declare RCX clobber (used as CL for shift count)", op);
        }
    }

    #[test]
    fn flag_setting_addsub_clobbers_r8_r11_for_setcc_dest() {
        let c = clobbers_for_op(Op::AddsFlags64);
        for reg in [GprMask::R8, GprMask::R9, GprMask::R10, GprMask::R11] {
            assert!(c.gpr.contains(reg),
                "AddsFlags must declare {:?} (used as setcc destination)", reg);
        }
    }

    #[test]
    fn loads_clobber_all_caller_saved_for_callback() {
        let c = clobbers_for_op(Op::Load64);
        assert!(c.gpr.contains(CALLER_SAVED_GPRS),
            "memory callbacks may clobber any caller-saved GPR");
    }
}
