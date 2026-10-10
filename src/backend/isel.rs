#![allow(clippy::too_many_arguments)]

use iced_x86::code_asm::*;

use crate::arch::{Cond, NUM_GPRS, ZR_ENCODING};
use crate::backend::abi::{
    ARG0_REG, ARG3_REG, CALL_PRECALL_SUB, CTX_REG, SCRATCH0, SCRATCH1, SCRATCH2, SCRATCH3,
};
use crate::backend::operand::{
    get_xmm_q, gpr8, gpr16, gpr32, gpr64, into_xmm_q, load_xmm_d, load_xmm_q, load_xmm_s, load32,
    load64, store_xmm_d, store_xmm_q, store_xmm_s, store32, store64, working_xmm_for,
};
use crate::backend::regalloc::{Allocation, Loc};
use crate::error::{Error, Result};
use crate::ir::{Armlet, Block, Op, Ty, ValueRef};
use crate::jit::context::cpu_offsets;

pub type EmitFn = fn(&mut CodeAssembler, &Block, &Allocation, usize) -> Result<()>;

#[inline]
fn dst_of(a: &Armlet, idx: usize) -> Option<ValueRef> {
    if a.ty != Ty::Void {
        Some(ValueRef::new(idx as u32))
    } else {
        None
    }
}

pub fn emit_armlet(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    if a.is_eliminated() {
        return Ok(());
    }
    if a.op.is_terminator() {
        return Ok(());
    }

    let f = dispatch_op(a.op).ok_or(Error::Unsupported {
        pc: block.start_pc,
        opcode: a.op as u32,
    })?;
    f(asm, block, alloc, idx)
}

fn dispatch_op(op: Op) -> Option<EmitFn> {
    use Op::*;
    Some(match op {
        Void => emit_nop,
        Identity => emit_op_identity,

        ConstU32 => emit_op_const_u32,
        ConstU64 => emit_op_const_u64,

        GetX => emit_op_get_x,
        GetW => emit_op_get_w,
        SetX => emit_op_set_x,
        SetW => emit_op_set_w,
        GetSp => emit_op_get_sp,
        SetSp => emit_op_set_sp,
        GetNzcv => emit_op_get_nzcv,
        SetNzcv => emit_op_set_nzcv,
        GetPc => emit_op_get_pc,
        GetV => emit_op_get_v,
        SetV => emit_op_set_v,

        Add32 | Add64 => emit_op_add,
        Sub32 | Sub64 => emit_op_sub,
        And32 | And64 => emit_op_and,
        Or32 | Or64 => emit_op_or,
        Eor32 | Eor64 => emit_op_xor,
        Mul32 | Mul64 => emit_op_mul,
        UMulH64 => emit_op_umulh,
        SMulH64 => emit_op_smulh,

        Adc32 | Adc64 => emit_op_adc,
        Sbc32 | Sbc64 => emit_op_sbc,
        Adcs32 | Adcs64 => emit_op_adcs,
        Sbcs32 | Sbcs64 => emit_op_sbcs,

        UDiv32 | UDiv64 => emit_op_udiv,
        SDiv32 | SDiv64 => emit_op_sdiv,

        Clz32 | Clz64 => emit_op_clz,
        Cls32 | Cls64 => emit_op_cls,
        Rbit32 | Rbit64 => emit_op_rbit,
        Rev16 => emit_op_rev16,
        Rev32 => emit_op_rev32,
        Rev64 => emit_op_rev64,
        Crc32Ieee8 | Crc32Ieee16 | Crc32Ieee32 | Crc32Ieee64
        | Crc32Castagnoli8 | Crc32Castagnoli16 | Crc32Castagnoli32 | Crc32Castagnoli64 => emit_op_crc32,

        Lsl32 | Lsl64 => emit_op_lsl,
        Lsr32 | Lsr64 => emit_op_lsr,
        Asr32 | Asr64 => emit_op_asr,
        Ror32 | Ror64 => emit_op_ror,

        Not32 | Not64 => emit_op_not,
        Neg32 | Neg64 => emit_op_neg,

        AddsFlags32 | AddsFlags64 | SubsFlags32 | SubsFlags64 => emit_op_flagged_addsub,

        Load8 | Load16 | Load32 | Load64 | Load128 => emit_op_load,
        Store8 | Store16 | Store32 | Store64 | Store128 => emit_op_store,

        LoadEx8 | LoadEx16 | LoadEx32 | LoadEx64 => emit_op_load_ex,
        StoreEx8 | StoreEx16 | StoreEx32 | StoreEx64 => emit_op_store_ex,

        Csel32 | Csel64 => emit_op_csel,

        Fadd32 | Fadd64 => emit_op_fadd,
        Fsub32 | Fsub64 => emit_op_fsub,
        Fmul32 | Fmul64 => emit_op_fmul,
        Fmadd32 | Fmadd64 => emit_op_scalar_fma,
        FrsqrtEstimate32 | FrsqrtEstimate64 | FrsqrtStep32 | FrsqrtStep64
        | FrecipEstimate32 | FrecipEstimate64 | FrecipStep32 | FrecipStep64 => emit_op_reciprocal,
        SaturatingNarrow => emit_op_saturating_narrow,
        FabsDiff32 | FabsDiff64 => emit_op_absolute_difference,
        FpMinMaxReduce32 => emit_op_fp_reduce,
        FpMinMaxNum32 | FpMinMaxNum64 => emit_op_fp_minmax_number,
        FpAbsoluteCompare32 | FpAbsoluteCompare64 => emit_op_absolute_compare,
        SaturatingBinary => emit_op_saturating_binary,
        IntMinMaxReduce32 => emit_op_integer_reduce,
        AesEncrypt | AesDecrypt | AesMixColumns | AesInverseMixColumns => emit_op_crypto_aes,
        Sha256Hash | Sha256Hash2 | Sha256Schedule0 | Sha256Schedule1 => emit_op_crypto_sha256,
        Fdiv32 | Fdiv64 => emit_op_fdiv,
        Fmax32 | Fmax64 => emit_op_fmax,
        Fmin32 | Fmin64 => emit_op_fmin,
        Fcmp32 | Fcmp64 => emit_op_fcmp,
        Fsqrt32 | Fsqrt64 => emit_op_fsqrt_,
        Fneg32 | Fneg64 => emit_op_fneg_,
        Fabs32 | Fabs64 => emit_op_fabs_,

        FcvtZsSW => emit_op_fcvt_zs_sw,
        FcvtZsSX => emit_op_fcvt_zs_sx,
        FcvtZsDW => emit_op_fcvt_zs_dw,
        FcvtZsDX => emit_op_fcvt_zs_dx,
        FcvtZuSW | FcvtZuSX | FcvtZuDW | FcvtZuDX => emit_op_fcvt_zu,
        ScvtfWS => emit_op_scvtf_ws,
        ScvtfXS => emit_op_scvtf_xs,
        ScvtfWD => emit_op_scvtf_wd,
        ScvtfXD => emit_op_scvtf_xd,
        UcvtfWS | UcvtfXS | UcvtfWD | UcvtfXD => emit_op_ucvtf,
        FcvtSD => emit_op_fcvt_sd,
        FcvtDS => emit_op_fcvt_ds,

        VecBuildQ => emit_op_vec_build_q,
        VecExtractLo64 => emit_op_vec_extract_lo64,
        VecExtractHi64 => emit_op_vec_extract_hi64,
        VecExtract8 => emit_op_vec_extract8,
        VecExtract16 => emit_op_vec_extract16,
        VecExtract32 => emit_op_vec_extract32,

        VecAdd8 | VecAdd16 | VecAdd32 | VecAdd64 => emit_op_vec_add,
        VecSub8 | VecSub16 | VecSub32 | VecSub64 => emit_op_vec_sub,
        VecAnd => emit_op_vec_and,
        VecOrr => emit_op_vec_orr,
        VecEor => emit_op_vec_eor,
        VecBic => emit_op_vec_bic,
        VecOrn => emit_op_vec_orn,

        VecNeg8 | VecNeg16 | VecNeg32 | VecNeg64 => emit_op_vec_neg,
        VecAbs8 | VecAbs16 | VecAbs32 => emit_op_vec_abs,
        VecNot => emit_op_vec_not,

        VecMul8 | VecMul16 | VecMul32 | VecMul64 => emit_op_vec_mul,

        VecShlImm8 | VecShlImm16 | VecShlImm32 | VecShlImm64 => emit_op_vec_shl_imm,
        VecUshrImm8 | VecUshrImm16 | VecUshrImm32 | VecUshrImm64 => emit_op_vec_ushr_imm,
        VecSshrImm8 | VecSshrImm16 | VecSshrImm32 | VecSshrImm64 => emit_op_vec_sshr_imm,
        VecUshl8 | VecUshl16 | VecUshl32 | VecUshl64
        | VecSshl8 | VecSshl16 | VecSshl32 | VecSshl64 => emit_op_vec_variable_shift,

        VecCmEq8 | VecCmEq16 | VecCmEq32 | VecCmEq64 => emit_op_vec_cmeq,
        VecCmGt8 | VecCmGt16 | VecCmGt32 | VecCmGt64 => emit_op_vec_cmgt,
        VecCmGe8 | VecCmGe16 | VecCmGe32 | VecCmGe64 => emit_op_vec_cmge,
        VecCmHi8 | VecCmHi16 | VecCmHi32 | VecCmHi64 => emit_op_vec_cmhi,
        VecCmHs8 | VecCmHs16 | VecCmHs32 | VecCmHs64 => emit_op_vec_cmhs,

        VecBit => emit_op_vec_bit,
        VecBif => emit_op_vec_bif,
        VecBsl => emit_op_vec_bsl,

        VecDupGpr8 | VecDupGpr16 | VecDupGpr32 | VecDupGpr64 => emit_op_vec_dup_gpr,
        VecInsGpr8 | VecInsGpr16 | VecInsGpr32 | VecInsGpr64 => emit_op_vec_ins_gpr,

        VecExt => emit_op_vec_ext,
        VecZip1_8 | VecZip1_16 | VecZip1_32 | VecZip1_64 => emit_op_vec_zip1,
        VecZip2_8 | VecZip2_16 | VecZip2_32 | VecZip2_64 => emit_op_vec_zip2,

        VecSmin8 | VecSmin16 | VecSmin32 => emit_op_vec_smin,
        VecSmax8 | VecSmax16 | VecSmax32 => emit_op_vec_smax,
        VecUmin8 | VecUmin16 | VecUmin32 => emit_op_vec_umin,
        VecUmax8 | VecUmax16 | VecUmax32 => emit_op_vec_umax,

        VecAddv32 => emit_op_vec_addv32,

        VecFAdd_S | VecFAdd_D => emit_op_vec_fadd,
        VecFSub_S | VecFSub_D => emit_op_vec_fsub,
        VecFMul_S | VecFMul_D => emit_op_vec_fmul,
        VecFDiv_S | VecFDiv_D => emit_op_vec_fdiv,
        VecFMax_S | VecFMax_D => emit_op_vec_fmax,
        VecFMin_S | VecFMin_D => emit_op_vec_fmin,
        VecFNeg_S | VecFNeg_D => emit_op_vec_fneg,
        VecFAbs_S | VecFAbs_D => emit_op_vec_fabs,
        VecFSqrt_S | VecFSqrt_D => emit_op_vec_fsqrt,
        VecFCmEq_S | VecFCmEq_D => emit_op_vec_fcmeq,
        VecFCmGt_S | VecFCmGt_D => emit_op_vec_fcmgt,
        VecFCmGe_S | VecFCmGe_D => emit_op_vec_fcmge,
        VecFmla_S | VecFmla_D => emit_op_vec_fmla,
        VecFmls_S | VecFmls_D => emit_op_vec_fmls,

        VecFRintN_S | VecFRintN_D => emit_op_vec_frintn,
        VecFRintM_S | VecFRintM_D => emit_op_vec_frintm,
        VecFRintP_S | VecFRintP_D => emit_op_vec_frintp,
        VecFRintZ_S | VecFRintZ_D => emit_op_vec_frintz,
        VecFRintA_S | VecFRintA_D => emit_op_vec_frinta,
        VecFRintX_S | VecFRintX_D => emit_op_vec_frintx,

        VecSaddl => emit_op_vec_addl_signed,
        VecUaddl => emit_op_vec_addl_unsigned,
        VecSsubl => emit_op_vec_subl_signed,
        VecUsubl => emit_op_vec_subl_unsigned,
        VecSmull => emit_op_vec_mull_signed,
        VecUmull => emit_op_vec_mull_unsigned,
        VecXtn => emit_op_vec_xtn,
        VecXtn2 => emit_op_vec_xtn2,
        VecTbl => emit_op_vec_tbl,
        VecTbl2 => emit_op_vec_tbl2,
        VecTbl3 => emit_op_vec_tbl3,
        VecRev16 => emit_op_vec_rev16,
        VecRev32 => emit_op_vec_rev32,
        VecRev64 => emit_op_vec_rev64,
        VecUzp1 => emit_op_vec_uzp1,
        VecUzp2 => emit_op_vec_uzp2,
        VecTrn1 => emit_op_vec_trn1,
        VecTrn2 => emit_op_vec_trn2,

        Hint | MemoryBarrier => emit_nop,
        Clrex => emit_op_clrex,

        Mrs => emit_op_mrs,
        Msr => emit_op_msr,

        _ => return None,
    })
}

fn emit_nop(_: &mut CodeAssembler, _: &Block, _: &Allocation, _: usize) -> Result<()> {
    Ok(())
}

fn emit_op_identity(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = match dst_of(&a, idx) {
        Some(d) => d,
        None => return Ok(()),
    };
    if alloc.loc(a.args[0]) == alloc.loc(d) {
        return Ok(());
    }
    if a.ty.bits() <= 32 {
        load32(asm, alloc, a.args[0], eax)?;
        store32(asm, alloc, d, eax)?;
    } else {
        load64(asm, alloc, a.args[0], SCRATCH0)?;
        store64(asm, alloc, d, SCRATCH0)?;
    }
    Ok(())
}

fn emit_op_const_u32(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    asm.mov(eax, (a.imm as u32) as i32)?;
    store32(asm, alloc, d, eax)?;
    Ok(())
}
fn emit_op_const_u64(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    asm.mov(SCRATCH0, a.imm as i64)?;
    store64(asm, alloc, d, SCRATCH0)?;
    Ok(())
}

fn emit_op_get_x(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    load_guest_x(asm, SCRATCH0, a.imm as usize)?;
    store64(asm, alloc, d, SCRATCH0)?;
    Ok(())
}
fn emit_op_get_w(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    load_guest_x(asm, SCRATCH0, a.imm as usize)?;
    store32(asm, alloc, d, eax)?;
    Ok(())
}
fn emit_op_set_x(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    load64(asm, alloc, a.args[0], SCRATCH0)?;
    store_guest_x(asm, a.imm as usize, SCRATCH0)?;
    Ok(())
}
fn emit_op_set_w(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    load32(asm, alloc, a.args[0], eax)?;
    // EAX is the value loaded above.  Using SCRATCH0 here left W-form
    // writes dependent on stale host state, corrupting the upper-level X
    // register and breaking startup code that mixes W and X arithmetic.
    store_guest_x(asm, a.imm as usize, rax)?;
    Ok(())
}
fn emit_op_get_sp(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    asm.mov(SCRATCH0, qword_ptr(CTX_REG + cpu_offsets::sp() as i32))?;
    store64(asm, alloc, d, SCRATCH0)?;
    Ok(())
}
fn emit_op_set_sp(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    _idx: usize,
) -> Result<()> {
    let a = block.code[_idx];
    load64(asm, alloc, a.args[0], SCRATCH0)?;
    asm.mov(qword_ptr(CTX_REG + cpu_offsets::sp() as i32), SCRATCH0)?;
    Ok(())
}
fn emit_op_get_nzcv(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    asm.movzx(eax, byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32))?;
    store32(asm, alloc, d, eax)?;
    Ok(())
}
fn emit_op_set_nzcv(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    load32(asm, alloc, a.args[0], eax)?;
    asm.mov(byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32), al)?;
    Ok(())
}
fn emit_op_get_pc(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    asm.mov(SCRATCH0, a.imm as i64)?;
    store64(asm, alloc, d, SCRATCH0)?;
    Ok(())
}

fn emit_op_get_v(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let off = cpu_offsets::vreg(a.imm as usize) as i32;
    match a.ty {
        Ty::U32 => {
            asm.mov(eax, dword_ptr(CTX_REG + off))?;
            store32(asm, alloc, d, eax)?;
        }
        Ty::U64 => {
            asm.mov(SCRATCH0, qword_ptr(CTX_REG + off))?;
            store64(asm, alloc, d, SCRATCH0)?;
        }
        Ty::U128 => {
            let dst_xmm = working_xmm_for(alloc, d, xmm0);
            asm.movdqu(dst_xmm, xmmword_ptr(CTX_REG + off))?;
            store_xmm_q(asm, alloc, d, dst_xmm)?;
        }
        other => {
            return Err(Error::Backend(format!(
                "GetV with unsupported ty {:?}",
                other
            )));
        }
    }
    Ok(())
}
fn emit_op_set_v(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let off = cpu_offsets::vreg(a.imm as usize) as i32;
    let src_ty = block.code[a.args[0].as_usize()].ty;
    match src_ty {
        Ty::U32 => {
            load32(asm, alloc, a.args[0], eax)?;
            asm.mov(dword_ptr(CTX_REG + off), eax)?;
            asm.mov(dword_ptr(CTX_REG + off + 4), 0i32)?;
            asm.mov(qword_ptr(CTX_REG + off + 8), 0i32)?;
        }
        Ty::U64 => {
            load64(asm, alloc, a.args[0], SCRATCH0)?;
            asm.mov(qword_ptr(CTX_REG + off), SCRATCH0)?;
            asm.mov(qword_ptr(CTX_REG + off + 8), 0i32)?;
        }
        Ty::U128 => {
            let src = get_xmm_q(asm, alloc, a.args[0], xmm0)?;
            asm.movdqu(xmmword_ptr(CTX_REG + off), src)?;
        }
        other => {
            return Err(Error::Backend(format!(
                "SetV with unsupported src ty {:?}",
                other
            )));
        }
    }
    Ok(())
}

macro_rules! adapt_binop {
    ($name:ident, $kind:expr) => {
        fn $name(
            asm: &mut CodeAssembler,
            block: &Block,
            alloc: &Allocation,
            idx: usize,
        ) -> Result<()> {
            let a = block.code[idx];
            emit_binop(asm, alloc, a, dst_of(&a, idx), $kind, a.op.size_bits())
        }
    };
}
adapt_binop!(emit_op_add, BinKind::Add);
adapt_binop!(emit_op_sub, BinKind::Sub);
adapt_binop!(emit_op_and, BinKind::And);
adapt_binop!(emit_op_or, BinKind::Or);
adapt_binop!(emit_op_xor, BinKind::Xor);
adapt_binop!(emit_op_mul, BinKind::Imul);

fn emit_op_umulh(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_mulh(asm, alloc, a, dst_of(&a, idx), false)
}
fn emit_op_smulh(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_mulh(asm, alloc, a, dst_of(&a, idx), true)
}

fn emit_mulh(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    signed: bool,
) -> Result<()> {
    load64(asm, alloc, a.args[0], rax)?;
    load64(asm, alloc, a.args[1], rcx)?;
    if signed {
        asm.imul(rcx)?;
    } else {
        asm.mul(rcx)?;
    }
    if let Some(d) = dst {
        store64(asm, alloc, d, rdx)?;
    } else {
        asm.nop()?;
    }
    Ok(())
}

macro_rules! adapt_adc {
    ($name:ident, $subtract:expr, $setflags:expr) => {
        fn $name(
            asm: &mut CodeAssembler,
            block: &Block,
            alloc: &Allocation,
            idx: usize,
        ) -> Result<()> {
            let a = block.code[idx];
            emit_adc_sbc(asm, alloc, a, dst_of(&a, idx), $subtract, a.op.size_bits(), $setflags)
        }
    };
}
adapt_adc!(emit_op_adc, false, false);
adapt_adc!(emit_op_sbc, true, false);
adapt_adc!(emit_op_adcs, false, true);
adapt_adc!(emit_op_sbcs, true, true);

macro_rules! adapt_div {
    ($name:ident, $signed:expr) => {
        fn $name(
            asm: &mut CodeAssembler,
            block: &Block,
            alloc: &Allocation,
            idx: usize,
        ) -> Result<()> {
            let a = block.code[idx];
            emit_div(asm, alloc, a, dst_of(&a, idx), $signed, a.op.size_bits())
        }
    };
}
adapt_div!(emit_op_udiv, false);
adapt_div!(emit_op_sdiv, true);

macro_rules! adapt_unop_count {
    ($name:ident, $emit:ident) => {
        fn $name(
            asm: &mut CodeAssembler,
            block: &Block,
            alloc: &Allocation,
            idx: usize,
        ) -> Result<()> {
            let a = block.code[idx];
            $emit(asm, alloc, a, dst_of(&a, idx), a.op.size_bits())
        }
    };
}
adapt_unop_count!(emit_op_clz, emit_clz);
adapt_unop_count!(emit_op_cls, emit_cls);
adapt_unop_count!(emit_op_rbit, emit_rbit);

fn emit_op_rev16(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let bits = if a.ty == Ty::U64 { 64 } else { 32 };
    emit_rev16(asm, alloc, a, dst_of(&a, idx), bits)
}
fn emit_op_rev32(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_rev32_within64(asm, alloc, a, dst_of(&a, idx))
}
fn emit_op_rev64(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let bits = if a.ty == Ty::U64 { 64 } else { 32 };
    emit_bswap(asm, alloc, a, dst_of(&a, idx), bits)
}

macro_rules! adapt_shift {
    ($name:ident, $kind:expr) => {
        fn $name(
            asm: &mut CodeAssembler,
            block: &Block,
            alloc: &Allocation,
            idx: usize,
        ) -> Result<()> {
            let a = block.code[idx];
            emit_shift(asm, alloc, a, dst_of(&a, idx), $kind, a.op.size_bits())
        }
    };
}
adapt_shift!(emit_op_lsl, ShiftKind::Lsl);
adapt_shift!(emit_op_lsr, ShiftKind::Lsr);
adapt_shift!(emit_op_asr, ShiftKind::Asr);
adapt_shift!(emit_op_ror, ShiftKind::Ror);

macro_rules! adapt_unop_simple {
    ($name:ident, $kind:expr) => {
        fn $name(
            asm: &mut CodeAssembler,
            block: &Block,
            alloc: &Allocation,
            idx: usize,
        ) -> Result<()> {
            let a = block.code[idx];
            emit_unop(asm, alloc, a, dst_of(&a, idx), $kind, a.op.size_bits())
        }
    };
}
adapt_unop_simple!(emit_op_not, UnopKind::Not);
adapt_unop_simple!(emit_op_neg, UnopKind::Neg);

fn emit_op_flagged_addsub(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_flagged_addsub(asm, alloc, a, dst_of(&a, idx))
}

fn emit_op_load(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let bytes = if matches!(a.op, Op::Load128) {
        16
    } else {
        a.op.size_bytes()
    };
    emit_load(asm, alloc, a, dst_of(&a, idx), bytes, block.use_fastmem)
}
fn emit_op_store(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let bytes = if matches!(a.op, Op::Store128) {
        16
    } else {
        a.op.size_bytes()
    };
    emit_store(asm, alloc, a, bytes, block.use_fastmem)
}
fn emit_op_load_ex(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_load_ex(asm, alloc, a, dst_of(&a, idx), a.op.size_bytes())
}
fn emit_op_store_ex(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_store_ex(asm, alloc, a, dst_of(&a, idx), a.op.size_bytes())
}

fn emit_op_csel(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_csel(asm, alloc, a, dst_of(&a, idx))
}

macro_rules! adapt_fbinop {
    ($name:ident, $kind:expr) => {
        fn $name(
            asm: &mut CodeAssembler,
            block: &Block,
            alloc: &Allocation,
            idx: usize,
        ) -> Result<()> {
            let a = block.code[idx];
            emit_fbinop(asm, alloc, a, dst_of(&a, idx), $kind, a.op.size_bits())
        }
    };
}
adapt_fbinop!(emit_op_fadd, FpBinKind::Add);
adapt_fbinop!(emit_op_fsub, FpBinKind::Sub);
adapt_fbinop!(emit_op_fmul, FpBinKind::Mul);
adapt_fbinop!(emit_op_fdiv, FpBinKind::Div);
adapt_fbinop!(emit_op_fmax, FpBinKind::Max);
adapt_fbinop!(emit_op_fmin, FpBinKind::Min);

fn emit_op_fcmp(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_fcmp(asm, alloc, a, a.op.size_bits())
}
fn emit_op_fsqrt_(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_fsqrt(asm, alloc, a, dst_of(&a, idx), a.op.size_bits())
}
fn emit_op_fneg_(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_fneg(asm, alloc, a, dst_of(&a, idx), a.op.size_bits())
}
fn emit_op_fabs_(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_fabs(asm, alloc, a, dst_of(&a, idx), a.op.size_bits())
}

macro_rules! adapt_fcvt_zs {
    ($name:ident, $double:expr, $to_x:expr) => {
        fn $name(
            asm: &mut CodeAssembler,
            block: &Block,
            alloc: &Allocation,
            idx: usize,
        ) -> Result<()> {
            let a = block.code[idx];
            emit_fcvt_zs(asm, alloc, a, dst_of(&a, idx), $double, $to_x)
        }
    };
}
adapt_fcvt_zs!(emit_op_fcvt_zs_sw, false, false);
adapt_fcvt_zs!(emit_op_fcvt_zs_sx, false, true);
adapt_fcvt_zs!(emit_op_fcvt_zs_dw, true, false);
adapt_fcvt_zs!(emit_op_fcvt_zs_dx, true, true);

macro_rules! adapt_scvtf {
    ($name:ident, $double:expr, $from_x:expr) => {
        fn $name(
            asm: &mut CodeAssembler,
            block: &Block,
            alloc: &Allocation,
            idx: usize,
        ) -> Result<()> {
            let a = block.code[idx];
            emit_scvtf(asm, alloc, a, dst_of(&a, idx), $double, $from_x)
        }
    };
}
adapt_scvtf!(emit_op_scvtf_ws, false, false);
adapt_scvtf!(emit_op_scvtf_xs, false, true);
adapt_scvtf!(emit_op_scvtf_wd, true, false);
adapt_scvtf!(emit_op_scvtf_xd, true, true);

fn emit_op_fcvt_sd(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_fcvt_precision(asm, alloc, a, dst_of(&a, idx), false)
}
fn emit_op_fcvt_ds(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_fcvt_precision(asm, alloc, a, dst_of(&a, idx), true)
}

fn emit_op_vec_build_q(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let working = working_xmm_for(alloc, d, xmm0);
    load64(asm, alloc, a.args[0], rax)?;
    asm.movq(working, rax)?;
    load64(asm, alloc, a.args[1], rax)?;
    asm.pinsrq(working, rax, 1)?;
    store_xmm_q(asm, alloc, d, working)?;
    Ok(())
}
fn emit_op_vec_extract_lo64(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src = get_xmm_q(asm, alloc, a.args[0], xmm0)?;
    asm.movq(rax, src)?;
    store64(asm, alloc, d, rax)
}
fn emit_op_vec_extract_hi64(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src = get_xmm_q(asm, alloc, a.args[0], xmm0)?;
    asm.pextrq(rax, src, 1)?;
    store64(asm, alloc, d, rax)
}
fn emit_op_vec_extract8(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src = get_xmm_q(asm, alloc, a.args[0], xmm0)?;
    asm.pextrb(eax, src, a.imm as i32)?;
    store32(asm, alloc, d, eax)
}
fn emit_op_vec_extract16(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src = get_xmm_q(asm, alloc, a.args[0], xmm0)?;
    asm.pextrw(eax, src, a.imm as i32)?;
    store32(asm, alloc, d, eax)
}
fn emit_op_vec_extract32(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src = get_xmm_q(asm, alloc, a.args[0], xmm0)?;
    asm.pextrd(eax, src, a.imm as i32)?;
    store32(asm, alloc, d, eax)
}

fn emit_op_vec_add(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_vec_binop(
        asm,
        alloc,
        a,
        dst_of(&a, idx),
        VecBinKind::Add(a.op.size_log2()),
    )
}
fn emit_op_vec_sub(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_vec_binop(
        asm,
        alloc,
        a,
        dst_of(&a, idx),
        VecBinKind::Sub(a.op.size_log2()),
    )
}
macro_rules! adapt_vec_logic {
    ($name:ident, $kind:expr) => {
        fn $name(
            asm: &mut CodeAssembler,
            block: &Block,
            alloc: &Allocation,
            idx: usize,
        ) -> Result<()> {
            let a = block.code[idx];
            emit_vec_binop(asm, alloc, a, dst_of(&a, idx), $kind)
        }
    };
}
adapt_vec_logic!(emit_op_vec_and, VecBinKind::And);
adapt_vec_logic!(emit_op_vec_orr, VecBinKind::Orr);
adapt_vec_logic!(emit_op_vec_eor, VecBinKind::Eor);
adapt_vec_logic!(emit_op_vec_bic, VecBinKind::Bic);
adapt_vec_logic!(emit_op_vec_orn, VecBinKind::Orn);

fn emit_op_vec_neg(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    asm.pxor(working, working)?;
    let src = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    match a.op.size_log2() {
        0 => asm.psubb(working, src)?,
        1 => asm.psubw(working, src)?,
        2 => asm.psubd(working, src)?,
        3 => asm.psubq(working, src)?,
        _ => unreachable!(),
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_abs(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    let src = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    match a.op.size_log2() {
        0 => asm.pabsb(working, src)?,
        1 => asm.pabsw(working, src)?,
        2 => asm.pabsd(working, src)?,
        _ => {
            return Err(Error::Backend(format!(
                "VecAbs lane {} not supported",
                a.op.size_log2()
            )));
        }
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_not(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    asm.pcmpeqd(xmm1, xmm1)?;
    asm.pxor(working, xmm1)?;
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_mul(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    match a.op.size_log2() {
        0 => {
            // x86 has no packed byte multiply. Odd bytes are isolated as
            // word low halves before PMULLW; even-byte products need only
            // their low8 bits, so cross terms are removed before merging.
            // Keep both original operands intact, especially when aliased.
            asm.movdqa(xmm2, working)?;
            asm.psrlw(xmm2, 8i32)?;
            asm.movdqa(xmm3, other)?;
            asm.psrlw(xmm3, 8i32)?;
            asm.pmullw(xmm2, xmm3)?;
            asm.psllw(xmm2, 8i32)?;
            asm.pmullw(working, other)?;
            asm.psllw(working, 8i32)?;
            asm.psrlw(working, 8i32)?;
            asm.por(working, xmm2)?;
        }
        1 => asm.pmullw(working, other)?,
        2 => asm.pmulld(working, other)?,
        3 => emit_mul64_into(asm, working, other)?,
        _ => {
            return Err(Error::Backend(format!(
                "VecMul lane {} not supported",
                a.op.size_log2()
            )));
        }
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_variable_shift(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let signed = a.op.base() == Op::VecSshl8.base();
    let bytes = a.op.size_bytes();
    let width = bytes * 8;
    let vector_bytes = if a.imm & 1 != 0 { 16 } else { 8 };
    // Keep both inputs untouched, even when all guest operands alias.
    into_xmm_q(asm, alloc, a.args[0], xmm1)?;
    into_xmm_q(asm, alloc, a.args[1], xmm2)?;
    asm.pxor(xmm0, xmm0)?;
    for lane in 0..vector_bytes / bytes {
        match bytes {
            1 => asm.pextrb(eax, xmm1, lane as i32)?,
            2 => asm.pextrw(eax, xmm1, lane as i32)?,
            4 => asm.pextrd(eax, xmm1, lane as i32)?,
            8 => asm.pextrq(rax, xmm1, lane as i32)?,
            _ => unreachable!(),
        }
        if signed {
            match bytes {
                1 => asm.movsx(rax, al)?,
                2 => asm.movsx(rax, ax)?,
                4 => asm.movsxd(rax, eax)?,
                _ => {},
            }
        }
        // Arm uses only the signed low byte, not the full element value.
        asm.pextrb(ecx, xmm2, (lane * bytes) as i32)?;
        asm.movsx(ecx, cl)?;
        let mut right = asm.create_label();
        let mut zero = asm.create_label();
        let mut insert = asm.create_label();
        asm.test(ecx, ecx)?;
        asm.js(right)?;
        asm.cmp(ecx, width as i32)?;
        asm.jae(zero)?;
        asm.shl(rax, cl)?;
        asm.jmp(insert)?;
        asm.set_label(&mut right)?;
        asm.neg(ecx)?;
        asm.cmp(ecx, width as i32)?;
        if signed {
            // Arithmetic right overshifts retain the original sign; clamping
            // to width-1 prevents x86's modulo-64 count from changing it.
            asm.mov(edx, (width - 1) as i32)?;
            asm.cmovae(ecx, edx)?;
            asm.sar(rax, cl)?;
        } else {
            asm.jae(zero)?;
            asm.shr(rax, cl)?;
        }
        asm.jmp(insert)?;
        asm.set_label(&mut zero)?;
        asm.xor(eax, eax)?;
        asm.set_label(&mut insert)?;
        match bytes {
            1 => asm.pinsrb(xmm0, eax, lane as i32)?,
            2 => asm.pinsrw(xmm0, eax, lane as i32)?,
            4 => asm.pinsrd(xmm0, eax, lane as i32)?,
            8 => asm.pinsrq(xmm0, rax, lane as i32)?,
            _ => unreachable!(),
        }
    }
    // Starting from zero also clears Q=0's unused upper 64 bits.
    store_xmm_q(asm, alloc, dst_of(&a, idx).unwrap(), xmm0)
}

fn broadcast_byte_mask(asm: &mut CodeAssembler, byte: u8) -> Result<()> {
    let pat = u64::from_le_bytes([byte; 8]) as i64;
    asm.mov(rax, pat)?;
    asm.movq(xmm1, rax)?;
    asm.punpcklqdq(xmm1, xmm1)?;
    Ok(())
}

fn emit_op_vec_shl_imm(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let shift = (a.imm >> 1) as u32;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    match a.op.size_log2() {
        0 => {
            asm.psllw(working, shift as i32)?;
            let mask_byte = ((0xFFu32 << shift) & 0xFF) as u8;
            broadcast_byte_mask(asm, mask_byte)?;
            asm.pand(working, xmm1)?;
        }
        1 => asm.psllw(working, shift as i32)?,
        2 => asm.pslld(working, shift as i32)?,
        3 => asm.psllq(working, shift as i32)?,
        _ => {
            return Err(Error::Backend(format!(
                "VecShlImm lane {} not supported",
                a.op.size_log2()
            )));
        }
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_ushr_imm(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let shift = (a.imm >> 1) as u32;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    match a.op.size_log2() {
        0 => {
            asm.psrlw(working, shift as i32)?;
            let mask_byte = (0xFFu32 >> shift) as u8;
            broadcast_byte_mask(asm, mask_byte)?;
            asm.pand(working, xmm1)?;
        }
        1 => asm.psrlw(working, shift as i32)?,
        2 => asm.psrld(working, shift as i32)?,
        3 => asm.psrlq(working, shift as i32)?,
        _ => {
            return Err(Error::Backend(format!(
                "VecUshrImm lane {} not supported",
                a.op.size_log2()
            )));
        }
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_sshr_imm(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let shift = (a.imm >> 1) as u32;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    match a.op.size_log2() {
        0 => {
            asm.pmovsxbw(xmm1, working)?;
            asm.psraw(xmm1, shift as i32)?;
            if q_form {
                asm.psrldq(working, 8)?;
                asm.pmovsxbw(xmm2, working)?;
                asm.psraw(xmm2, shift as i32)?;
                asm.packsswb(xmm1, xmm2)?;
            } else {
                asm.packsswb(xmm1, xmm1)?;
            }
            if working != xmm1 {
                asm.movdqa(working, xmm1)?;
            }
        }
        1 => asm.psraw(working, shift as i32)?,
        2 => asm.psrad(working, shift as i32)?,
        3 => {
            asm.pxor(xmm1, xmm1)?;
            emit_pcmpgtq_sse41(asm, xmm1, working)?;
            asm.psllq(xmm1, (64 - shift) as i32)?;
            asm.psrlq(working, shift as i32)?;
            asm.por(working, xmm1)?;
        }
        _ => unreachable!(),
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

/// SSE4.1 has no packed signed 64-bit compare (PCMPGTQ is SSE4.2).  Keep the
/// baseline at SSE4.1 by comparing the high and low 32-bit halves after
/// flipping their sign bits.  XMM4..XMM7 are scratch-only registers in the
/// allocator; preserve them so spilled vector values remain live.
fn emit_pcmpgtq_sse41(
    asm: &mut CodeAssembler,
    dst: AsmRegisterXmm,
    src: AsmRegisterXmm,
) -> Result<()> {
    asm.sub(rsp, 96)?;
    for (i, reg) in [xmm4, xmm5, xmm6, xmm7].into_iter().enumerate() {
        asm.movdqu(xmmword_ptr(rsp + (i as i32) * 16), reg)?;
    }
    asm.movdqu(xmmword_ptr(rsp + 64), dst)?;
    asm.movdqu(xmmword_ptr(rsp + 80), src)?;

    // High 32-bit signed compare (after sign-bit flip) and equality mask.
    asm.movdqa(xmm4, dst)?;
    asm.movdqa(xmm5, src)?;
    asm.psrlq(xmm4, 32)?;
    asm.psrlq(xmm5, 32)?;
    asm.movdqa(xmm7, xmm4)?;
    asm.pcmpeqd(xmm7, xmm5)?;
    asm.pcmpgtd(xmm4, xmm5)?;
    asm.pshufd(xmm4, xmm4, 0xA0)?;
    asm.pshufd(xmm7, xmm7, 0xA0)?;

    // Sign-bit mask used only for the low-half unsigned comparison.
    asm.pcmpeqd(xmm6, xmm6)?;
    asm.pslld(xmm6, 31)?;

    // Low 32-bit unsigned compare, again using a sign-bit flip.  Duplicate
    // each low-dword result across its 64-bit lane.
    asm.movdqu(dst, xmmword_ptr(rsp + 64))?;
    asm.movdqu(xmm5, xmmword_ptr(rsp + 80))?;
    asm.pxor(dst, xmm6)?;
    asm.pxor(xmm5, xmm6)?;
    asm.pcmpgtd(dst, xmm5)?;
    asm.pshufd(dst, dst, 0xA0)?;
    asm.pand(xmm7, dst)?;
    asm.por(xmm4, xmm7)?;
    asm.movdqa(dst, xmm4)?;

    for (i, reg) in [xmm4, xmm5, xmm6, xmm7].into_iter().enumerate() {
        asm.movdqu(reg, xmmword_ptr(rsp + (i as i32) * 16))?;
    }
    asm.add(rsp, 96)?;
    Ok(())
}

fn emit_op_vec_cmeq(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    match a.op.size_log2() {
        0 => asm.pcmpeqb(working, other)?,
        1 => asm.pcmpeqw(working, other)?,
        2 => asm.pcmpeqd(working, other)?,
        3 => asm.pcmpeqq(working, other)?,
        _ => unreachable!(),
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_cmgt(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    match a.op.size_log2() {
        0 => asm.pcmpgtb(working, other)?,
        1 => asm.pcmpgtw(working, other)?,
        2 => asm.pcmpgtd(working, other)?,
        3 => emit_pcmpgtq_sse41(asm, working, other)?,
        _ => unreachable!(),
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_cmge(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[1], working)?;
    let vn = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    match a.op.size_log2() {
        0 => asm.pcmpgtb(working, vn)?,
        1 => asm.pcmpgtw(working, vn)?,
        2 => asm.pcmpgtd(working, vn)?,
        3 => emit_pcmpgtq_sse41(asm, working, vn)?,
        _ => unreachable!(),
    }
    asm.pcmpeqd(xmm2, xmm2)?;
    asm.pxor(working, xmm2)?;
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_unsigned_cmp(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    d: ValueRef,
    invert: bool,
) -> Result<()> {
    let q_form = (a.imm & 1) != 0;
    let lane = a.op.size_log2();
    let working = working_xmm_for(alloc, d, xmm0);

    if lane == 0 {
        let (sub_lhs, sub_rhs) = if invert {
            (a.args[1], a.args[0])
        } else {
            (a.args[0], a.args[1])
        };
        into_xmm_q(asm, alloc, sub_lhs, working)?;
        let rhs = get_xmm_q(asm, alloc, sub_rhs, xmm1)?;
        asm.psubusb(working, rhs)?;
        asm.pxor(xmm2, xmm2)?;
        asm.pcmpeqb(working, xmm2)?;
        if !invert {
            asm.pcmpeqd(xmm1, xmm1)?;
            asm.pxor(working, xmm1)?;
        }
        if !q_form {
            asm.movq(working, working)?;
        }
        return store_xmm_q(asm, alloc, d, working);
    }

    asm.pcmpeqd(xmm2, xmm2)?;
    let lane_bits_minus_1: i32 = (8 << lane) - 1;
    match lane {
        1 => asm.psllw(xmm2, lane_bits_minus_1)?,
        2 => asm.pslld(xmm2, lane_bits_minus_1)?,
        3 => asm.psllq(xmm2, lane_bits_minus_1)?,
        _ => {
            return Err(Error::Backend(format!(
                "unsigned cmp lane {} not supported",
                lane
            )));
        }
    }

    let (work_src, other_src) = if invert {
        (a.args[1], a.args[0])
    } else {
        (a.args[0], a.args[1])
    };
    into_xmm_q(asm, alloc, work_src, working)?;
    asm.pxor(working, xmm2)?;

    let other_src_xmm = get_xmm_q(asm, alloc, other_src, xmm1)?;
    if other_src_xmm != xmm1 {
        asm.movdqa(xmm1, other_src_xmm)?;
    }
    asm.pxor(xmm1, xmm2)?;

    match lane {
        1 => asm.pcmpgtw(working, xmm1)?,
        2 => asm.pcmpgtd(working, xmm1)?,
        3 => emit_pcmpgtq_sse41(asm, working, xmm1)?,
        _ => unreachable!(),
    }
    if invert {
        asm.pcmpeqd(xmm2, xmm2)?;
        asm.pxor(working, xmm2)?;
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_cmhi(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_unsigned_cmp(asm, alloc, a, d, false)
}
fn emit_op_vec_cmhs(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_unsigned_cmp(asm, alloc, a, d, true)
}

fn emit_op_vec_bit(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[2], working)?;
    let vn = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    asm.movdqa(xmm2, working)?;
    asm.pand(xmm2, vn)?;
    let vd = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    asm.pandn(working, vd)?;
    asm.por(working, xmm2)?;
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_bif(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[2], working)?;
    let vd = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    asm.movdqa(xmm2, working)?;
    asm.pand(xmm2, vd)?;
    let vn = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    asm.pandn(working, vn)?;
    asm.por(working, xmm2)?;
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_bsl(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    let vn = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    asm.movdqa(xmm2, working)?;
    asm.pand(xmm2, vn)?;
    let vm = get_xmm_q(asm, alloc, a.args[2], xmm1)?;
    asm.pandn(working, vm)?;
    asm.por(working, xmm2)?;
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_dup_gpr(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let lane = a.op.size_log2();
    let working = working_xmm_for(alloc, d, xmm0);

    match lane {
        0 => {
            load32(asm, alloc, a.args[0], eax)?;
            asm.movd(working, eax)?;
            asm.pxor(xmm1, xmm1)?;
            asm.pshufb(working, xmm1)?;
        }
        1 => {
            load32(asm, alloc, a.args[0], eax)?;
            asm.movd(working, eax)?;
            asm.pshuflw(working, working, 0)?;
            asm.pshufd(working, working, 0)?;
        }
        2 => {
            load32(asm, alloc, a.args[0], eax)?;
            asm.movd(working, eax)?;
            asm.pshufd(working, working, 0)?;
        }
        3 => {
            load64(asm, alloc, a.args[0], rax)?;
            asm.movq(working, rax)?;
            asm.punpcklqdq(working, working)?;
        }
        _ => unreachable!(),
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_ext(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let byte_off = (a.imm >> 1) as i32;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[1], working)?;
    let vn = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    asm.palignr(working, vn, byte_off)?;
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_zip1(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    match a.op.size_log2() {
        0 => asm.punpcklbw(working, other)?,
        1 => asm.punpcklwd(working, other)?,
        2 => asm.punpckldq(working, other)?,
        3 => asm.punpcklqdq(working, other)?,
        _ => unreachable!(),
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_zip2(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    if q_form {
        match a.op.size_log2() {
            0 => asm.punpckhbw(working, other)?,
            1 => asm.punpckhwd(working, other)?,
            2 => asm.punpckhdq(working, other)?,
            3 => asm.punpckhqdq(working, other)?,
            _ => unreachable!(),
        }
    } else {
        asm.psrldq(working, 4)?;
        asm.movdqa(xmm2, other)?;
        asm.psrldq(xmm2, 4)?;
        match a.op.size_log2() {
            0 => asm.punpcklbw(working, xmm2)?,
            1 => asm.punpcklwd(working, xmm2)?,
            2 => asm.punpckldq(working, xmm2)?,
            _ => return Err(Error::Backend("ZIP2 64-bit lane requires Q form".into())),
        }
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

macro_rules! emit_vec_minmax {
    ($fn_name:ident, $b:ident, $w:ident, $d:ident, $kind:ident) => {
        fn $fn_name(
            asm: &mut CodeAssembler,
            block: &Block,
            alloc: &Allocation,
            idx: usize,
        ) -> Result<()> {
            let a = block.code[idx];
            let d = dst_of(&a, idx).unwrap();
            let q_form = (a.imm & 1) != 0;
            let working = working_xmm_for(alloc, d, xmm0);
            into_xmm_q(asm, alloc, a.args[0], working)?;
            let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
            match a.op.size_log2() {
                0 => asm.$b(working, other)?,
                1 => asm.$w(working, other)?,
                2 => asm.$d(working, other)?,
                3 => emit_minmax64(asm, working, other, MinMaxKind::$kind)?,
                _ => unreachable!(),
            }
            if !q_form {
                asm.movq(working, working)?;
            }
            store_xmm_q(asm, alloc, d, working)
        }
    };
}
emit_vec_minmax!(emit_op_vec_smin, pminsb, pminsw, pminsd, Smin);
emit_vec_minmax!(emit_op_vec_smax, pmaxsb, pmaxsw, pmaxsd, Smax);
emit_vec_minmax!(emit_op_vec_umin, pminub, pminuw, pminud, Umin);
emit_vec_minmax!(emit_op_vec_umax, pmaxub, pmaxuw, pmaxud, Umax);

#[derive(Clone, Copy)]
enum MinMaxKind {
    Smin,
    Smax,
    Umin,
    Umax,
}

fn emit_minmax64(
    asm: &mut CodeAssembler,
    working: AsmRegisterXmm,
    other: AsmRegisterXmm,
    kind: MinMaxKind,
) -> Result<()> {
    let unsigned = matches!(kind, MinMaxKind::Umin | MinMaxKind::Umax);
    let is_max = matches!(kind, MinMaxKind::Smax | MinMaxKind::Umax);

    if other != xmm1 {
        asm.movdqa(xmm1, other)?;
    }

    if unsigned {
        asm.pcmpeqd(xmm3, xmm3)?;
        asm.psllq(xmm3, 63)?;
        asm.pxor(working, xmm3)?;
        asm.pxor(xmm1, xmm3)?;
    }

    asm.movdqa(xmm2, if is_max { xmm1 } else { working })?;
    if is_max {
        emit_pcmpgtq_sse41(asm, xmm2, working)?;
    } else {
        emit_pcmpgtq_sse41(asm, xmm2, xmm1)?;
    }

    asm.movdqa(xmm3, working)?;
    asm.pxor(xmm3, xmm1)?;
    asm.pand(xmm3, xmm2)?;
    asm.pxor(working, xmm3)?;

    if unsigned {
        asm.pcmpeqd(xmm3, xmm3)?;
        asm.psllq(xmm3, 63)?;
        asm.pxor(working, xmm3)?;
    }
    Ok(())
}

#[inline]
fn vec_fp_is_double(op: Op) -> bool {
    (op as u16 & 1) != 0
}

macro_rules! emit_vec_fbin {
    ($name:ident, $ps:ident, $pd:ident) => {
        fn $name(
            asm: &mut CodeAssembler,
            block: &Block,
            alloc: &Allocation,
            idx: usize,
        ) -> Result<()> {
            let a = block.code[idx];
            let d = dst_of(&a, idx).unwrap();
            let q_form = (a.imm & 1) != 0;
            let double = vec_fp_is_double(a.op);
            let working = working_xmm_for(alloc, d, xmm0);
            into_xmm_q(asm, alloc, a.args[0], working)?;
            let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
            if double {
                asm.$pd(working, other)?;
            } else {
                asm.$ps(working, other)?;
            }
            if !q_form {
                asm.movq(working, working)?;
            }
            store_xmm_q(asm, alloc, d, working)
        }
    };
}
emit_vec_fbin!(emit_op_vec_fadd, addps, addpd);
emit_vec_fbin!(emit_op_vec_fsub, subps, subpd);
emit_vec_fbin!(emit_op_vec_fmul, mulps, mulpd);
emit_vec_fbin!(emit_op_vec_fdiv, divps, divpd);
emit_vec_fbin!(emit_op_vec_fmax, maxps, maxpd);
emit_vec_fbin!(emit_op_vec_fmin, minps, minpd);

fn emit_op_vec_fneg(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let double = vec_fp_is_double(a.op);
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    asm.pcmpeqd(xmm1, xmm1)?;
    if double {
        asm.psllq(xmm1, 63)?;
    } else {
        asm.pslld(xmm1, 31)?;
    }
    asm.pxor(working, xmm1)?;
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_fabs(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let double = vec_fp_is_double(a.op);
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    asm.pcmpeqd(xmm1, xmm1)?;
    if double {
        asm.psrlq(xmm1, 1)?;
    } else {
        asm.psrld(xmm1, 1)?;
    }
    asm.pand(working, xmm1)?;
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_fsqrt(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let double = vec_fp_is_double(a.op);
    let working = working_xmm_for(alloc, d, xmm0);
    let src = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    if double {
        asm.sqrtpd(working, src)?;
    } else {
        asm.sqrtps(working, src)?;
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_vec_frint_fixed(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    d: ValueRef,
    mode: i32,
) -> Result<()> {
    let q_form = (a.imm & 1) != 0;
    let double = vec_fp_is_double(a.op);
    let working = working_xmm_for(alloc, d, xmm0);
    let src = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    let imm = mode | 0x08;
    if double {
        asm.roundpd(working, src, imm)?;
    } else {
        asm.roundps(working, src, imm)?;
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_frintn(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_vec_frint_fixed(asm, alloc, a, d, 0)
}
fn emit_op_vec_frintm(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_vec_frint_fixed(asm, alloc, a, d, 1)
}
fn emit_op_vec_frintp(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_vec_frint_fixed(asm, alloc, a, d, 2)
}
fn emit_op_vec_frintz(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_vec_frint_fixed(asm, alloc, a, d, 3)
}
fn emit_op_vec_frintx(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_vec_frint_fixed(asm, alloc, a, d, 0)
}

fn emit_op_vec_frinta(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let double = vec_fp_is_double(a.op);
    let working = working_xmm_for(alloc, d, xmm0);

    into_xmm_q(asm, alloc, a.args[0], working)?;

    asm.pcmpeqd(xmm2, xmm2)?;
    if double {
        asm.psllq(xmm2, 63)?;
    } else {
        asm.pslld(xmm2, 31)?;
    }
    asm.pand(xmm2, working)?;

    asm.pcmpeqd(xmm1, xmm1)?;
    if double {
        asm.psrlq(xmm1, 1)?;
    } else {
        asm.psrld(xmm1, 1)?;
    }
    asm.pand(working, xmm1)?;

    // Adding 0.5 before truncation misrounds the immediate predecessor of a
    // tie when the addition rounds upward. Truncate first, then compare the
    // exact fractional part to 0.5 and conditionally add one instead.
    asm.movdqa(xmm1, working)?;
    if double {
        asm.roundpd(working, working, 0x0B)?;
        asm.subpd(xmm1, working)?;
    } else {
        asm.roundps(working, working, 0x0B)?;
        asm.subps(xmm1, working)?;
    }

    if double {
        asm.mov(rax, 0x3FE0_0000_0000_0000u64 as i64)?;
    } else {
        asm.mov(rax, 0x3F00_0000_3F00_0000u64 as i64)?;
    }
    asm.movq(xmm3, rax)?;
    asm.punpcklqdq(xmm3, xmm3)?;
    if double {
        asm.cmppd(xmm3, xmm1, 2)?; // Ordered 0.5 <= fractional magnitude.
        asm.mov(rax, 0x3FF0_0000_0000_0000u64 as i64)?;
    } else {
        asm.cmpps(xmm3, xmm1, 2)?;
        asm.mov(rax, 0x3F80_0000_3F80_0000u64 as i64)?;
    }
    asm.movq(xmm1, rax)?;
    asm.punpcklqdq(xmm1, xmm1)?;
    asm.pand(xmm3, xmm1)?;
    if double {
        asm.addpd(working, xmm3)?;
    } else {
        asm.addps(working, xmm3)?;
    }
    asm.por(working, xmm2)?;

    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_fcmeq(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let double = vec_fp_is_double(a.op);
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    if double {
        asm.cmppd(working, other, 0)?;
    } else {
        asm.cmpps(working, other, 0)?;
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_fcmgt(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let double = vec_fp_is_double(a.op);
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[1], working)?;
    let other = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    if double {
        asm.cmppd(working, other, 1)?;
    } else {
        asm.cmpps(working, other, 1)?;
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_fcmge(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let double = vec_fp_is_double(a.op);
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[1], working)?;
    let other = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    if double {
        asm.cmppd(working, other, 2)?;
    } else {
        asm.cmpps(working, other, 2)?;
    }
    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_fma_inner(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    d: ValueRef,
    subtract: bool,
) -> Result<()> {
    let q_form = (a.imm & 1) != 0;
    let double = vec_fp_is_double(a.op);
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    let vn = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    let vm = get_xmm_q(asm, alloc, a.args[2], xmm2)?;

    let fused = crate::backend::cpu_features::active_features().has_avx
        && crate::backend::cpu_features::active_features().has_fma;
    if fused {
        if subtract {
            asm.movdqa(xmm3, vn)?;
            if double {
                asm.cmpunordpd(xmm3, xmm3)?;
                asm.psllq(xmm3, 63)?;
            } else {
                asm.cmpunordps(xmm3, xmm3)?;
                asm.pslld(xmm3, 31)?;
            }
        }
        match (subtract, double) {
            (false, false) => asm.vfmadd231ps(working, vn, vm)?,
            (false, true) => asm.vfmadd231pd(working, vn, vm)?,
            (true, false) => asm.vfnmadd231ps(working, vn, vm)?,
            (true, true) => asm.vfnmadd231pd(working, vn, vm)?,
        }
        if subtract {
            asm.pxor(working, xmm3)?;
        }
    } else {
        // SSE4.1-only fallback. It sacrifices fused rounding, but remains
        // legal on the baseline host and preserves architectural state.
        asm.movdqa(xmm3, vn)?;
        if double {
            asm.mulpd(xmm3, vm)?;
            if subtract {
                asm.subpd(working, xmm3)?;
            } else {
                asm.addpd(working, xmm3)?;
            }
        } else {
            asm.mulps(xmm3, vm)?;
            if subtract {
                asm.subps(working, xmm3)?;
            } else {
                asm.addps(working, xmm3)?;
            }
        }
        if subtract {
            asm.movdqa(xmm3, working)?;
            if double {
                asm.cmpunordpd(xmm3, xmm3)?;
                asm.psllq(xmm3, 63)?;
            } else {
                asm.cmpunordps(xmm3, xmm3)?;
                asm.pslld(xmm3, 31)?;
            }
            asm.pxor(working, xmm3)?;
        }
    }

    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_fmla(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_fma_inner(asm, alloc, a, d, false)
}
fn emit_op_vec_fmls(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_fma_inner(asm, alloc, a, d, true)
}

fn emit_mul64_into(
    asm: &mut CodeAssembler,
    working: AsmRegisterXmm,
    other: AsmRegisterXmm,
) -> Result<()> {
    if other != xmm1 {
        asm.movdqa(xmm1, other)?;
    }
    asm.pshufd(xmm2, working, 0xB1)?;
    asm.pmuludq(xmm2, xmm1)?;
    asm.pshufd(xmm3, xmm1, 0xB1)?;
    asm.pmuludq(xmm3, working)?;
    asm.paddq(xmm2, xmm3)?;
    asm.psllq(xmm2, 32)?;
    asm.pmuludq(working, xmm1)?;
    asm.paddq(working, xmm2)?;
    Ok(())
}

fn emit_op_vec_addl_signed(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_widening_op(asm, alloc, a, d, true, WideningOp::Add)
}
fn emit_op_vec_addl_unsigned(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_widening_op(asm, alloc, a, d, false, WideningOp::Add)
}

#[derive(Clone, Copy)]
enum WideningOp {
    Add,
    Sub,
    Mul,
}

fn emit_widening_op(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    d: ValueRef,
    signed: bool,
    op: WideningOp,
) -> Result<()> {
    let high_half = ((a.imm >> 1) & 1) != 0;
    let src_lane = ((a.imm >> 2) & 0x3) as u32;
    let working = working_xmm_for(alloc, d, xmm0);

    into_xmm_q(asm, alloc, a.args[0], working)?;
    let other_src = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    if other_src != xmm1 {
        asm.movdqa(xmm1, other_src)?;
    }

    if high_half {
        asm.psrldq(working, 8)?;
        asm.psrldq(xmm1, 8)?;
    }

    if signed {
        match src_lane {
            0 => {
                asm.pmovsxbw(working, working)?;
                asm.pmovsxbw(xmm1, xmm1)?;
            }
            1 => {
                asm.pmovsxwd(working, working)?;
                asm.pmovsxwd(xmm1, xmm1)?;
            }
            2 => {
                asm.pmovsxdq(working, working)?;
                asm.pmovsxdq(xmm1, xmm1)?;
            }
            _ => {
                return Err(Error::Backend(format!(
                    "widening signed lane {} unsupported",
                    src_lane
                )));
            }
        }
    } else {
        match src_lane {
            0 => {
                asm.pmovzxbw(working, working)?;
                asm.pmovzxbw(xmm1, xmm1)?;
            }
            1 => {
                asm.pmovzxwd(working, working)?;
                asm.pmovzxwd(xmm1, xmm1)?;
            }
            2 => {
                asm.pmovzxdq(working, working)?;
                asm.pmovzxdq(xmm1, xmm1)?;
            }
            _ => {
                return Err(Error::Backend(format!(
                    "widening unsigned lane {} unsupported",
                    src_lane
                )));
            }
        }
    }

    match op {
        WideningOp::Add => match src_lane {
            0 => asm.paddw(working, xmm1)?,
            1 => asm.paddd(working, xmm1)?,
            2 => asm.paddq(working, xmm1)?,
            _ => unreachable!(),
        },
        WideningOp::Sub => match src_lane {
            0 => asm.psubw(working, xmm1)?,
            1 => asm.psubd(working, xmm1)?,
            2 => asm.psubq(working, xmm1)?,
            _ => unreachable!(),
        },
        WideningOp::Mul => match src_lane {
            0 => asm.pmullw(working, xmm1)?,
            1 => asm.pmulld(working, xmm1)?,
            2 => emit_mul64_into(asm, working, xmm1)?,
            _ => unreachable!(),
        },
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_subl_signed(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_widening_op(asm, alloc, a, d, true, WideningOp::Sub)
}
fn emit_op_vec_subl_unsigned(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_widening_op(asm, alloc, a, d, false, WideningOp::Sub)
}
fn emit_op_vec_mull_signed(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_widening_op(asm, alloc, a, d, true, WideningOp::Mul)
}
fn emit_op_vec_mull_unsigned(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_widening_op(asm, alloc, a, d, false, WideningOp::Mul)
}

fn emit_rev_with_mask(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    d: ValueRef,
    mask_lo: u64,
    mask_hi: u64,
) -> Result<()> {
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;

    asm.mov(rax, mask_lo as i64)?;
    asm.movq(xmm1, rax)?;
    asm.mov(rax, mask_hi as i64)?;
    asm.pinsrq(xmm1, rax, 1)?;
    asm.pshufb(working, xmm1)?;

    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_rev16(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src_lane = ((a.imm >> 2) & 0x3) as u32;
    if src_lane != 0 {
        return Err(Error::Backend(format!(
            "REV16 only valid for B lanes (got log2={})",
            src_lane
        )));
    }
    emit_rev_with_mask(
        asm,
        alloc,
        a,
        d,
        0x0607_0405_0203_0001,
        0x0E0F_0C0D_0A0B_0809,
    )
}

fn emit_op_vec_rev32(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src_lane = ((a.imm >> 2) & 0x3) as u32;
    let (lo, hi) = match src_lane {
        0 => (0x0405_0607_0001_0203, 0x0C0D_0E0F_0809_0A0B),
        1 => (0x0504_0706_0100_0302, 0x0D0C_0F0E_0908_0B0A),
        _ => {
            return Err(Error::Backend(format!(
                "REV32 invalid src_lane {}",
                src_lane
            )));
        }
    };
    emit_rev_with_mask(asm, alloc, a, d, lo, hi)
}

#[derive(Clone, Copy)]
enum PermKind {
    Uzp1,
    Uzp2,
    Trn1,
    Trn2,
}

fn perm_masks(kind: PermKind, lane_log2: u32, q_form: bool) -> (u64, u64, u64, u64) {
    let lane_bytes = 1usize << lane_log2;
    let num_result_lanes = (if q_form { 16 } else { 8 }) / lane_bytes;
    let half = num_result_lanes / 2;

    let mut mask_n = [0x80u8; 16];
    let mut mask_m = [0x80u8; 16];

    for r in 0..num_result_lanes {
        let (use_vm, src_lane) = match kind {
            PermKind::Uzp1 => {
                if r < half {
                    (false, r * 2)
                } else {
                    (true, (r - half) * 2)
                }
            }
            PermKind::Uzp2 => {
                if r < half {
                    (false, r * 2 + 1)
                } else {
                    (true, (r - half) * 2 + 1)
                }
            }
            PermKind::Trn1 => ((r & 1) == 1, r & !1),
            PermKind::Trn2 => ((r & 1) == 1, (r & !1) + 1),
        };
        let mask = if use_vm { &mut mask_m } else { &mut mask_n };
        for b in 0..lane_bytes {
            mask[r * lane_bytes + b] = (src_lane * lane_bytes + b) as u8;
        }
    }

    let to_u64 = |slice: &[u8]| -> u64 {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(slice);
        u64::from_le_bytes(buf)
    };
    (
        to_u64(&mask_n[..8]),
        to_u64(&mask_n[8..]),
        to_u64(&mask_m[..8]),
        to_u64(&mask_m[8..]),
    )
}

fn emit_uzp_trn(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    d: ValueRef,
    kind: PermKind,
) -> Result<()> {
    let q_form = (a.imm & 1) != 0;
    let lane_log2 = ((a.imm >> 2) & 0x3) as u32;

    if lane_log2 == 3 {
        let working = working_xmm_for(alloc, d, xmm0);
        into_xmm_q(asm, alloc, a.args[0], working)?;
        let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
        match kind {
            PermKind::Uzp1 | PermKind::Trn1 => asm.punpcklqdq(working, other)?,
            PermKind::Uzp2 | PermKind::Trn2 => asm.punpckhqdq(working, other)?,
        }
        if !q_form {
            asm.movq(working, working)?;
        }
        return store_xmm_q(asm, alloc, d, working);
    }

    let (n_lo, n_hi, m_lo, m_hi) = perm_masks(kind, lane_log2, q_form);
    let working = working_xmm_for(alloc, d, xmm0);

    into_xmm_q(asm, alloc, a.args[0], working)?;
    asm.mov(rax, n_lo as i64)?;
    asm.movq(xmm1, rax)?;
    asm.mov(rax, n_hi as i64)?;
    asm.pinsrq(xmm1, rax, 1)?;
    asm.pshufb(working, xmm1)?;

    let vm_src = get_xmm_q(asm, alloc, a.args[1], xmm2)?;
    if vm_src != xmm2 {
        asm.movdqa(xmm2, vm_src)?;
    }
    asm.mov(rax, m_lo as i64)?;
    asm.movq(xmm1, rax)?;
    asm.mov(rax, m_hi as i64)?;
    asm.pinsrq(xmm1, rax, 1)?;
    asm.pshufb(xmm2, xmm1)?;

    asm.por(working, xmm2)?;

    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_uzp1(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_uzp_trn(asm, alloc, a, d, PermKind::Uzp1)
}
fn emit_op_vec_uzp2(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_uzp_trn(asm, alloc, a, d, PermKind::Uzp2)
}
fn emit_op_vec_trn1(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_uzp_trn(asm, alloc, a, d, PermKind::Trn1)
}
fn emit_op_vec_trn2(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_uzp_trn(asm, alloc, a, d, PermKind::Trn2)
}

fn emit_op_vec_rev64(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src_lane = ((a.imm >> 2) & 0x3) as u32;
    let (lo, hi) = match src_lane {
        0 => (0x0001_0203_0405_0607, 0x0809_0A0B_0C0D_0E0F),
        1 => (0x0100_0302_0504_0706, 0x0908_0B0A_0D0C_0F0E),
        2 => (0x0302_0100_0706_0504, 0x0B0A_0908_0F0E_0D0C),
        _ => {
            return Err(Error::Backend(format!(
                "REV64 invalid src_lane {}",
                src_lane
            )));
        }
    };
    emit_rev_with_mask(asm, alloc, a, d, lo, hi)
}

fn emit_op_vec_tbl(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);

    asm.mov(rax, 0x7070_7070_7070_7070u64 as i64)?;
    asm.movq(xmm2, rax)?;
    asm.punpcklqdq(xmm2, xmm2)?;

    into_xmm_q(asm, alloc, a.args[1], xmm1)?;
    asm.paddusb(xmm1, xmm2)?;

    into_xmm_q(asm, alloc, a.args[0], working)?;
    asm.pshufb(working, xmm1)?;

    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_broadcast_byte(asm: &mut CodeAssembler, dst: AsmRegisterXmm, byte: u8) -> Result<()> {
    let pat = u64::from_le_bytes([byte; 8]) as i64;
    asm.mov(rax, pat)?;
    asm.movq(dst, rax)?;
    asm.punpcklqdq(dst, dst)?;
    Ok(())
}

fn emit_tbl_chunk_or(
    asm: &mut CodeAssembler,
    working: AsmRegisterXmm,
    table_xmm: AsmRegisterXmm,
    indices_xmm: AsmRegisterXmm,
    chunk_offset: u8,
    pad_70: AsmRegisterXmm,
    scratch_idx: AsmRegisterXmm,
    scratch_res: AsmRegisterXmm,
) -> Result<()> {
    asm.movdqa(scratch_idx, indices_xmm)?;
    if chunk_offset > 0 {
        emit_broadcast_byte(asm, scratch_res, chunk_offset)?;
        asm.psubusb(scratch_idx, scratch_res)?;
    }
    asm.paddusb(scratch_idx, pad_70)?;

    asm.movdqa(scratch_res, table_xmm)?;
    asm.pshufb(scratch_res, scratch_idx)?;

    if chunk_offset == 0 {
        asm.por(working, scratch_res)?;
    } else {
        asm.movdqa(scratch_idx, pad_70)?;
        emit_broadcast_byte(asm, scratch_idx, chunk_offset)?;
        asm.psubusb(scratch_idx, indices_xmm)?;
        asm.pxor(pad_70, pad_70)?;
        asm.pcmpeqb(scratch_idx, pad_70)?;
        emit_broadcast_byte(asm, pad_70, 0x70)?;
        asm.pand(scratch_res, scratch_idx)?;
        asm.por(working, scratch_res)?;
    }
    Ok(())
}

fn emit_op_vec_tbl2(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);

    into_xmm_q(asm, alloc, a.args[2], xmm1)?;

    emit_broadcast_byte(asm, xmm2, 0x70)?;

    asm.pxor(working, working)?;

    let t0 = get_xmm_q(asm, alloc, a.args[0], xmm3)?;
    if t0 != xmm3 {
        asm.movdqa(xmm3, t0)?;
    }
    emit_tbl_chunk_or(asm, working, xmm3, xmm1, 0, xmm2, xmm4, xmm5)?;
    let t1 = get_xmm_q(asm, alloc, a.args[1], xmm3)?;
    if t1 != xmm3 {
        asm.movdqa(xmm3, t1)?;
    }
    emit_tbl_chunk_or(asm, working, xmm3, xmm1, 16, xmm2, xmm4, xmm5)?;

    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_tbl3(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);

    into_xmm_q(asm, alloc, a.args[3], xmm1)?;
    emit_broadcast_byte(asm, xmm2, 0x70)?;
    asm.pxor(working, working)?;

    for (i, table_arg) in [a.args[0], a.args[1], a.args[2]].iter().enumerate() {
        let off = (i * 16) as u8;
        let t = get_xmm_q(asm, alloc, *table_arg, xmm3)?;
        if t != xmm3 {
            asm.movdqa(xmm3, t)?;
        }
        emit_tbl_chunk_or(asm, working, xmm3, xmm1, off, xmm2, xmm4, xmm5)?;
    }

    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_narrow_into(
    asm: &mut CodeAssembler,
    src_xmm: AsmRegisterXmm,
    dst_xmm: AsmRegisterXmm,
    src_lane: u32,
) -> Result<()> {
    if src_xmm != dst_xmm {
        asm.movdqa(dst_xmm, src_xmm)?;
    }
    match src_lane {
        1 => {
            asm.pcmpeqd(xmm1, xmm1)?;
            asm.psrlw(xmm1, 8)?;
            asm.pand(dst_xmm, xmm1)?;
            asm.packuswb(dst_xmm, dst_xmm)?;
        }
        2 => {
            asm.pcmpeqd(xmm1, xmm1)?;
            asm.psrld(xmm1, 16)?;
            asm.pand(dst_xmm, xmm1)?;
            asm.packusdw(dst_xmm, dst_xmm)?;
        }
        3 => {
            asm.pshufd(dst_xmm, dst_xmm, 0x08)?;
        }
        _ => {
            return Err(Error::Backend(format!(
                "XTN src lane {} not supported",
                src_lane
            )));
        }
    }
    Ok(())
}

fn emit_op_vec_xtn(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src_lane = ((a.imm >> 2) & 0x3) as u32;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    emit_narrow_into(asm, working, working, src_lane)?;
    asm.movq(working, working)?;
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_xtn2(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src_lane = ((a.imm >> 2) & 0x3) as u32;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    asm.movq(working, working)?;
    let vn_src = get_xmm_q(asm, alloc, a.args[1], xmm2)?;
    if vn_src != xmm2 {
        asm.movdqa(xmm2, vn_src)?;
    }
    emit_narrow_into(asm, xmm2, xmm2, src_lane)?;
    asm.movq(xmm2, xmm2)?;
    asm.pslldq(xmm2, 8)?;
    asm.por(working, xmm2)?;
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_addv32(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    into_xmm_q(asm, alloc, a.args[0], xmm0)?;
    asm.phaddd(xmm0, xmm0)?;
    asm.phaddd(xmm0, xmm0)?;
    asm.movd(eax, xmm0)?;
    store32(asm, alloc, d, eax)
}

fn emit_op_vec_ins_gpr(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let lane_idx = (a.imm >> 1) as i32;
    let lane = a.op.size_log2();
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    match lane {
        0 => {
            load32(asm, alloc, a.args[1], eax)?;
            asm.pinsrb(working, eax, lane_idx)?;
        }
        1 => {
            load32(asm, alloc, a.args[1], eax)?;
            asm.pinsrw(working, eax, lane_idx)?;
        }
        2 => {
            load32(asm, alloc, a.args[1], eax)?;
            asm.pinsrd(working, eax, lane_idx)?;
        }
        3 => {
            load64(asm, alloc, a.args[1], rax)?;
            asm.pinsrq(working, rax, lane_idx)?;
        }
        _ => unreachable!(),
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_clrex(
    asm: &mut CodeAssembler,
    _block: &Block,
    _alloc: &Allocation,
    _idx: usize,
) -> Result<()> {
    asm.mov(
        byte_ptr(CTX_REG + cpu_offsets::exclusive_size() as i32),
        0i32,
    )?;
    Ok(())
}
fn emit_op_mrs(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_mrs(asm, alloc, a, dst_of(&a, idx))
}
fn emit_op_msr(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    emit_msr(asm, alloc, a)
}

fn load_guest_x(asm: &mut CodeAssembler, dst: AsmRegister64, reg: usize) -> Result<()> {
    if reg == ZR_ENCODING as usize {
        asm.xor(dst, dst)?;
        return Ok(());
    }
    debug_assert!(reg < NUM_GPRS);
    let off = cpu_offsets::xreg(reg) as i32;
    asm.mov(dst, qword_ptr(CTX_REG + off))?;
    Ok(())
}

fn store_guest_x(asm: &mut CodeAssembler, reg: usize, src: AsmRegister64) -> Result<()> {
    if reg == ZR_ENCODING as usize {
        return Ok(());
    }
    debug_assert!(reg < NUM_GPRS);
    let off = cpu_offsets::xreg(reg) as i32;
    asm.mov(qword_ptr(CTX_REG + off), src)?;
    Ok(())
}

#[derive(Clone, Copy)]
enum BinKind {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Imul,
}

fn apply_bin_32(
    asm: &mut CodeAssembler,
    k: BinKind,
    l: AsmRegister32,
    r: AsmRegister32,
) -> Result<()> {
    match k {
        BinKind::Add => asm.add(l, r)?,
        BinKind::Sub => asm.sub(l, r)?,
        BinKind::And => asm.and(l, r)?,
        BinKind::Or => asm.or(l, r)?,
        BinKind::Xor => asm.xor(l, r)?,
        BinKind::Imul => asm.imul_2(l, r)?,
    }
    Ok(())
}

fn apply_bin_64(
    asm: &mut CodeAssembler,
    k: BinKind,
    l: AsmRegister64,
    r: AsmRegister64,
) -> Result<()> {
    match k {
        BinKind::Add => asm.add(l, r)?,
        BinKind::Sub => asm.sub(l, r)?,
        BinKind::And => asm.and(l, r)?,
        BinKind::Or => asm.or(l, r)?,
        BinKind::Xor => asm.xor(l, r)?,
        BinKind::Imul => asm.imul_2(l, r)?,
    }
    Ok(())
}

fn emit_binop(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    k: BinKind,
    bits: u32,
) -> Result<()> {
    if bits == 64 {
        if let Some(d) = dst {
            if let Loc::Reg(r) = alloc.loc(d) {
                if alloc.loc(a.args[0]) == Loc::Reg(r) {
                    load64(asm, alloc, a.args[1], SCRATCH1)?;
                    apply_bin_64(asm, k, gpr64(r), SCRATCH1)?;
                    return Ok(());
                }
            }
        }
        load64(asm, alloc, a.args[0], SCRATCH0)?;
        load64(asm, alloc, a.args[1], SCRATCH1)?;
        apply_bin_64(asm, k, SCRATCH0, SCRATCH1)?;
        if let Some(d) = dst {
            store64(asm, alloc, d, SCRATCH0)?;
        }
    } else {
        if let Some(d) = dst {
            if let Loc::Reg(r) = alloc.loc(d) {
                if alloc.loc(a.args[0]) == Loc::Reg(r) {
                    load32(asm, alloc, a.args[1], gpr32(scratch1_id()))?;
                    apply_bin_32(asm, k, gpr32(r), gpr32(scratch1_id()))?;
                    return Ok(());
                }
            }
        }
        load32(asm, alloc, a.args[0], eax)?;
        load32(asm, alloc, a.args[1], gpr32(scratch1_id()))?;
        apply_bin_32(asm, k, eax, gpr32(scratch1_id()))?;
        if let Some(d) = dst {
            store32(asm, alloc, d, eax)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum UnopKind {
    Not,
    Neg,
}

fn emit_unop(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    k: UnopKind,
    bits: u32,
) -> Result<()> {
    if bits == 64 {
        load64(asm, alloc, a.args[0], SCRATCH0)?;
        match k {
            UnopKind::Not => asm.not(SCRATCH0)?,
            UnopKind::Neg => asm.neg(SCRATCH0)?,
        }
        if let Some(d) = dst {
            store64(asm, alloc, d, SCRATCH0)?;
        }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        match k {
            UnopKind::Not => asm.not(eax)?,
            UnopKind::Neg => asm.neg(eax)?,
        }
        if let Some(d) = dst {
            store32(asm, alloc, d, eax)?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum ShiftKind {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

fn emit_shift(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    kind: ShiftKind,
    bits: u32,
) -> Result<()> {
    if bits == 64 {
        load64(asm, alloc, a.args[0], SCRATCH0)?;
        load64(asm, alloc, a.args[1], rcx)?;
        match kind {
            ShiftKind::Lsl => asm.shl(SCRATCH0, cl)?,
            ShiftKind::Lsr => asm.shr(SCRATCH0, cl)?,
            ShiftKind::Asr => asm.sar(SCRATCH0, cl)?,
            ShiftKind::Ror => asm.ror(SCRATCH0, cl)?,
        }
        if let Some(d) = dst {
            store64(asm, alloc, d, SCRATCH0)?;
        }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        load32(asm, alloc, a.args[1], ecx)?;
        match kind {
            ShiftKind::Lsl => asm.shl(eax, cl)?,
            ShiftKind::Lsr => asm.shr(eax, cl)?,
            ShiftKind::Asr => asm.sar(eax, cl)?,
            ShiftKind::Ror => asm.ror(eax, cl)?,
        }
        if let Some(d) = dst {
            store32(asm, alloc, d, eax)?;
        }
    }
    Ok(())
}

fn emit_flagged_addsub(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
) -> Result<()> {
    let is_64 = matches!(a.op, Op::AddsFlags64 | Op::SubsFlags64);
    let is_sub = matches!(a.op, Op::SubsFlags32 | Op::SubsFlags64);

    if is_64 {
        load64(asm, alloc, a.args[0], SCRATCH0)?;
        load64(asm, alloc, a.args[1], SCRATCH1)?;
        if is_sub {
            asm.sub(SCRATCH0, SCRATCH1)?;
        } else {
            asm.add(SCRATCH0, SCRATCH1)?;
        }
        if let Some(d) = dst {
            store64(asm, alloc, d, SCRATCH0)?;
        }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        load32(asm, alloc, a.args[1], gpr32(scratch1_id()))?;
        if is_sub {
            asm.sub(eax, gpr32(scratch1_id()))?;
        } else {
            asm.add(eax, gpr32(scratch1_id()))?;
        }
        if let Some(d) = dst {
            store32(asm, alloc, d, eax)?;
        }
    }

    asm.sets(r8b)?;
    asm.sete(r9b)?;
    asm.setc(r10b)?;
    asm.seto(r11b)?;
    if is_sub {
        asm.xor(r10b, 1i32)?;
    }
    asm.movzx(eax, r8b)?;
    asm.shl(eax, 3i32)?;
    asm.movzx(ecx, r9b)?;
    asm.shl(ecx, 2i32)?;
    asm.or(eax, ecx)?;
    asm.movzx(ecx, r10b)?;
    asm.shl(ecx, 1i32)?;
    asm.or(eax, ecx)?;
    asm.movzx(ecx, r11b)?;
    asm.or(eax, ecx)?;
    asm.mov(byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32), al)?;
    Ok(())
}

fn emit_adc_sbc(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    is_sub: bool,
    bits: u32,
    setflags: bool,
) -> Result<()> {
    // Consume the explicit incoming NZCV SSA value. The optimizer may forward
    // it and eliminate a prior SetNzcv when this instruction overwrites flags.
    load32(asm, alloc, a.args[2], edx)?;
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        load64(asm, alloc, a.args[1], rcx)?;
        asm.bt(edx, 1i32)?;
        if is_sub { asm.cmc()?; }
        if is_sub {
            asm.sbb(rax, rcx)?;
        } else {
            asm.adc(rax, rcx)?;
        }
        if setflags {
            asm.sets(r8b)?;
            asm.sete(r9b)?;
            asm.setc(r10b)?;
            asm.seto(r11b)?;
        }
        if let Some(d) = dst {
            store64(asm, alloc, d, rax)?;
        }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        load32(asm, alloc, a.args[1], ecx)?;
        asm.bt(edx, 1i32)?;
        if is_sub { asm.cmc()?; }
        if is_sub {
            asm.sbb(eax, ecx)?;
        } else {
            asm.adc(eax, ecx)?;
        }
        if setflags {
            asm.sets(r8b)?;
            asm.sete(r9b)?;
            asm.setc(r10b)?;
            asm.seto(r11b)?;
        }
        if let Some(d) = dst {
            store32(asm, alloc, d, eax)?;
        }
    }
    if setflags {
        // SBB reports borrow; Arm AddWithCarry(N, NOT(M), C) reports no-borrow.
        if is_sub { asm.xor(r10b, 1i32)?; }
        asm.movzx(eax, r8b)?;
        asm.shl(eax, 3i32)?;
        asm.movzx(ecx, r9b)?;
        asm.shl(ecx, 2i32)?;
        asm.or(eax, ecx)?;
        asm.movzx(ecx, r10b)?;
        asm.shl(ecx, 1i32)?;
        asm.or(eax, ecx)?;
        asm.movzx(ecx, r11b)?;
        asm.or(eax, ecx)?;
        asm.mov(byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32), al)?;
    }
    Ok(())
}

fn emit_op_crypto_aes(
    asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let destination = dst_of(&a, idx).unwrap();
    into_xmm_q(asm, alloc, a.args[0], xmm0)?;
    if crate::backend::cpu_features::active_features().has_aesni {
        if a.op == Op::AesInverseMixColumns {
            asm.aesimc(xmm0, xmm0)?;
        } else {
            asm.pxor(xmm1, xmm1)?;
            match a.op {
                Op::AesEncrypt => asm.aesenclast(xmm0, xmm1)?,
                Op::AesDecrypt => asm.aesdeclast(xmm0, xmm1)?,
                Op::AesMixColumns => {
                    asm.aesdeclast(xmm0, xmm1)?;
                    asm.aesenc(xmm0, xmm1)?;
                }
                _ => unreachable!(),
            }
        }
    } else {
        let mode = match a.op {
            Op::AesEncrypt => 0,
            Op::AesDecrypt => 1,
            Op::AesMixColumns => 2,
            Op::AesInverseMixColumns => 3,
            _ => unreachable!(),
        };
        asm.sub(rsp, 32)?;
        asm.movdqu(xmmword_ptr(rsp), xmm0)?;
        asm.lea(ARG0_REG, ptr(rsp))?;
        asm.lea(SCRATCH1, ptr(rsp + 16))?;
        asm.mov(SCRATCH3, mode as i64)?;
        asm.sub(rsp, CALL_PRECALL_SUB)?;
        asm.mov(rax, crate::backend::crypto_aes::transform as *const () as usize as i64)?;
        asm.call(rax)?;
        asm.add(rsp, CALL_PRECALL_SUB)?;
        asm.movdqu(xmm0, xmmword_ptr(rsp + 16))?;
        asm.add(rsp, 32)?;
    }
    store_xmm_q(asm, alloc, destination, xmm0)
}

fn emit_op_crypto_sha256(
    asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let destination = dst_of(&a, idx).unwrap();
    asm.sub(rsp, 64)?;
    for index in 0..2 {
        into_xmm_q(asm, alloc, a.args[index], xmm0)?;
        asm.movdqu(xmmword_ptr(rsp + (index * 16) as i32), xmm0)?;
    }
    if a.op == Op::Sha256Schedule0 {
        asm.pxor(xmm0, xmm0)?;
    } else {
        into_xmm_q(asm, alloc, a.args[2], xmm0)?;
    }
    asm.movdqu(xmmword_ptr(rsp + 32), xmm0)?;
    let mode = match a.op {
        Op::Sha256Hash => 0,
        Op::Sha256Hash2 => 1,
        Op::Sha256Schedule0 => 2,
        Op::Sha256Schedule1 => 3,
        _ => unreachable!(),
    };
    let helper = if crate::backend::cpu_features::active_features().has_shani {
        crate::backend::crypto_sha256::accelerated as *const ()
    } else {
        crate::backend::crypto_sha256::portable as *const ()
    };
    asm.lea(ARG0_REG, ptr(rsp))?;
    asm.lea(SCRATCH1, ptr(rsp + 48))?;
    asm.mov(SCRATCH3, mode as i64)?;
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.mov(rax, helper as usize as i64)?;
    asm.call(rax)?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    asm.movdqu(xmm0, xmmword_ptr(rsp + 48))?;
    asm.add(rsp, 64)?;
    store_xmm_q(asm, alloc, destination, xmm0)
}

fn emit_op_saturating_binary(
    asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    load64(asm, alloc, a.args[0], SCRATCH1)?;
    load64(asm, alloc, a.args[1], SCRATCH3)?;
    load64(asm, alloc, a.args[2], ARG3_REG)?;
    #[cfg(target_os = "windows")]
    {
        load64(asm, alloc, a.args[3], rax)?;
        asm.mov(ARG0_REG, CTX_REG)?;
        // Four integer register arguments, 32-byte shadow space, two stack
        // arguments, and the existing 8-byte call-alignment adjustment.
        asm.sub(rsp, CALL_PRECALL_SUB + 16)?;
        asm.mov(qword_ptr(rsp + 32), rax)?;
        asm.mov(qword_ptr(rsp + 40), a.imm as i32)?;
    }
    #[cfg(not(target_os = "windows"))]
    {
        load64(asm, alloc, a.args[3], r8)?;
        asm.mov(ARG0_REG, CTX_REG)?;
        asm.mov(r9, a.imm as i64)?;
        asm.sub(rsp, CALL_PRECALL_SUB)?;
    }
    asm.mov(rax, crate::backend::simd_binary_status::saturating_binary as *const () as usize as i64)?;
    asm.call(rax)?;
    #[cfg(target_os = "windows")]
    asm.add(rsp, CALL_PRECALL_SUB + 16)?;
    #[cfg(not(target_os = "windows"))]
    asm.add(rsp, CALL_PRECALL_SUB)?;
    if let Some(destination) = dst_of(&a, idx) {
        asm.movdqu(xmm0, xmmword_ptr(CTX_REG + cpu_offsets::io_value() as i32))?;
        store_xmm_q(asm, alloc, destination, xmm0)?;
    }
    Ok(())
}

fn emit_op_absolute_compare(
    asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let double = a.op == Op::FpAbsoluteCompare64;
    if double {
        load64(asm, alloc, a.args[0], SCRATCH1)?;
        load64(asm, alloc, a.args[1], SCRATCH3)?;
    } else {
        #[cfg(target_os = "windows")]
        let (n_reg, m_reg) = (edx, r8d);
        #[cfg(not(target_os = "windows"))]
        let (n_reg, m_reg) = (esi, edx);
        load32(asm, alloc, a.args[0], n_reg)?;
        load32(asm, alloc, a.args[1], m_reg)?;
    }
    asm.mov(ARG0_REG, CTX_REG)?;
    asm.mov(ARG3_REG, a.imm as i64)?;
    asm.mov(rax, crate::backend::simd_binary_status::absolute_compare as *const () as usize as i64)?;
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.call(rax)?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    if let Some(destination) = dst_of(&a, idx) {
        if double { store64(asm, alloc, destination, rax)?; }
        else { store32(asm, alloc, destination, eax)?; }
    }
    Ok(())
}

fn emit_div(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    signed: bool,
    bits: u32,
) -> Result<()> {
    let mut lbl_zero = asm.create_label();
    let mut lbl_done = asm.create_label();

    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        load64(asm, alloc, a.args[1], rcx)?;
        asm.test(rcx, rcx)?;
        asm.jz(lbl_zero)?;
        if signed {
            let mut lbl_do_div = asm.create_label();
            asm.cmp(rcx, -1i32)?;
            asm.jne(lbl_do_div)?;
            asm.mov(rdx, i64::MIN)?;
            asm.cmp(rax, rdx)?;
            asm.je(lbl_done)?;
            asm.set_label(&mut lbl_do_div)?;
            asm.cqo()?;
            asm.idiv(rcx)?;
        } else {
            asm.xor(rdx, rdx)?;
            asm.div(rcx)?;
        }
        asm.jmp(lbl_done)?;
        asm.set_label(&mut lbl_zero)?;
        asm.xor(rax, rax)?;
        asm.set_label(&mut lbl_done)?;
        if let Some(d) = dst {
            store64(asm, alloc, d, rax)?;
        } else {
            asm.nop()?;
        }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        load32(asm, alloc, a.args[1], ecx)?;
        asm.test(ecx, ecx)?;
        asm.jz(lbl_zero)?;
        if signed {
            let mut lbl_do_div = asm.create_label();
            asm.cmp(ecx, -1i32)?;
            asm.jne(lbl_do_div)?;
            asm.cmp(eax, i32::MIN)?;
            asm.je(lbl_done)?;
            asm.set_label(&mut lbl_do_div)?;
            asm.cdq()?;
            asm.idiv(ecx)?;
        } else {
            asm.xor(edx, edx)?;
            asm.div(ecx)?;
        }
        asm.jmp(lbl_done)?;
        asm.set_label(&mut lbl_zero)?;
        asm.xor(eax, eax)?;
        asm.set_label(&mut lbl_done)?;
        if let Some(d) = dst {
            store32(asm, alloc, d, eax)?;
        } else {
            asm.nop()?;
        }
    }
    Ok(())
}

fn emit_clz(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bits: u32,
) -> Result<()> {
    let has_lzcnt = crate::backend::cpu_features::active_features().has_lzcnt;
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        if has_lzcnt {
            asm.lzcnt(rax, rax)?;
        } else {
            clz64_bsr(asm)?;
        }
        if let Some(d) = dst {
            store64(asm, alloc, d, rax)?;
        }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        if has_lzcnt {
            asm.lzcnt(eax, eax)?;
        } else {
            clz32_bsr(asm)?;
        }
        if let Some(d) = dst {
            store32(asm, alloc, d, eax)?;
        }
    }
    Ok(())
}

fn clz64_bsr(asm: &mut CodeAssembler) -> Result<()> {
    asm.bsr(rcx, rax)?;
    asm.mov(rax, 127i64)?;
    asm.cmovnz(rax, rcx)?;
    asm.xor(rax, 63i32)?;
    Ok(())
}
fn clz32_bsr(asm: &mut CodeAssembler) -> Result<()> {
    asm.bsr(ecx, eax)?;
    asm.mov(eax, 63i32)?;
    asm.cmovnz(eax, ecx)?;
    asm.xor(eax, 31i32)?;
    Ok(())
}

fn emit_cls(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bits: u32,
) -> Result<()> {
    let has_lzcnt = crate::backend::cpu_features::active_features().has_lzcnt;
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        asm.mov(rcx, rax)?;
        asm.sar(rcx, 1i32)?;
        asm.xor(rax, rcx)?;
        if has_lzcnt {
            asm.lzcnt(rax, rax)?;
        } else {
            clz64_bsr(asm)?;
        }
        asm.dec(rax)?;
        if let Some(d) = dst {
            store64(asm, alloc, d, rax)?;
        }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        asm.mov(ecx, eax)?;
        asm.sar(ecx, 1i32)?;
        asm.xor(eax, ecx)?;
        if has_lzcnt {
            asm.lzcnt(eax, eax)?;
        } else {
            clz32_bsr(asm)?;
        }
        asm.dec(eax)?;
        if let Some(d) = dst {
            store32(asm, alloc, d, eax)?;
        }
    }
    Ok(())
}

fn emit_rbit(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bits: u32,
) -> Result<()> {
    let has_gfni = crate::backend::cpu_features::active_features().has_gfni;
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        if has_gfni {
            rbit64_gfni(asm)?;
        } else {
            rbit64_pshufb(asm)?;
        }
        if let Some(d) = dst {
            store64(asm, alloc, d, rax)?;
        }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        if has_gfni {
            rbit32_gfni(asm)?;
        } else {
            rbit32_pshufb(asm)?;
        }
        if let Some(d) = dst {
            store32(asm, alloc, d, eax)?;
        }
    }
    Ok(())
}

fn rbit64_gfni(asm: &mut CodeAssembler) -> Result<()> {
    asm.movq(xmm0, rax)?;
    asm.mov(rcx, 0x8040201008040201u64 as i64)?;
    asm.movq(xmm1, rcx)?;
    asm.gf2p8affineqb(xmm0, xmm1, 0u32)?;
    asm.movq(rax, xmm0)?;
    asm.bswap(rax)?;
    Ok(())
}

fn rbit32_gfni(asm: &mut CodeAssembler) -> Result<()> {
    asm.movd(xmm0, eax)?;
    asm.mov(rcx, 0x8040201008040201u64 as i64)?;
    asm.movq(xmm1, rcx)?;
    asm.gf2p8affineqb(xmm0, xmm1, 0u32)?;
    asm.movd(eax, xmm0)?;
    asm.bswap(eax)?;
    Ok(())
}

#[repr(C, align(16))]
struct RbitConsts {
    hi_mask: [u64; 2],
    hi_rev_table: [u64; 2],
    lo_rev_table: [u64; 2],
}

static RBIT_CONSTS: RbitConsts = RbitConsts {
    hi_mask: [0xF0F0_F0F0_F0F0_F0F0; 2],
    hi_rev_table: [0xE060_A020_C040_8000, 0xF070_B030_D050_9010],
    lo_rev_table: [0x0E06_0A02_0C04_0800, 0x0F07_0B03_0D05_0901],
};

fn rbit64_pshufb(asm: &mut CodeAssembler) -> Result<()> {
    asm.movq(xmm0, rax)?;
    asm.mov(rcx, &RBIT_CONSTS as *const RbitConsts as i64)?;
    asm.movdqa(xmm1, xmmword_ptr(rcx))?;
    asm.movdqa(xmm2, xmmword_ptr(rcx + 16))?;
    asm.movdqa(xmm3, xmmword_ptr(rcx + 32))?;
    asm.pand(xmm1, xmm0)?;
    asm.pxor(xmm0, xmm1)?;
    asm.psrld(xmm1, 4i32)?;
    asm.pshufb(xmm2, xmm0)?;
    asm.pshufb(xmm3, xmm1)?;
    asm.por(xmm3, xmm2)?;
    asm.movq(rax, xmm3)?;
    asm.bswap(rax)?;
    Ok(())
}

fn rbit32_pshufb(asm: &mut CodeAssembler) -> Result<()> {
    asm.movd(xmm0, eax)?;
    asm.mov(rcx, &RBIT_CONSTS as *const RbitConsts as i64)?;
    asm.movdqa(xmm1, xmmword_ptr(rcx))?;
    asm.movdqa(xmm2, xmmword_ptr(rcx + 16))?;
    asm.movdqa(xmm3, xmmword_ptr(rcx + 32))?;
    asm.pand(xmm1, xmm0)?;
    asm.pxor(xmm0, xmm1)?;
    asm.psrld(xmm1, 4i32)?;
    asm.pshufb(xmm2, xmm0)?;
    asm.pshufb(xmm3, xmm1)?;
    asm.por(xmm3, xmm2)?;
    asm.movd(eax, xmm3)?;
    asm.bswap(eax)?;
    Ok(())
}

fn emit_rev16(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bits: u32,
) -> Result<()> {
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        asm.mov(rcx, rax)?;
        asm.shr(rcx, 8i32)?;
        asm.mov(rdx, 0x00FF_00FF_00FF_00FFi64)?;
        asm.and(rcx, rdx)?;
        asm.and(rax, rdx)?;
        asm.shl(rax, 8i32)?;
        asm.or(rax, rcx)?;
        if let Some(d) = dst {
            store64(asm, alloc, d, rax)?;
        }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        asm.mov(ecx, eax)?;
        asm.shr(ecx, 8i32)?;
        asm.and(ecx, 0x00FF_00FF_u32 as i32)?;
        asm.and(eax, 0x00FF_00FF_u32 as i32)?;
        asm.shl(eax, 8i32)?;
        asm.or(eax, ecx)?;
        if let Some(d) = dst {
            store32(asm, alloc, d, eax)?;
        }
    }
    Ok(())
}

fn emit_rev32_within64(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
) -> Result<()> {
    load64(asm, alloc, a.args[0], rax)?;
    asm.bswap(rax)?;
    asm.rol(rax, 32i32)?;
    if let Some(d) = dst {
        store64(asm, alloc, d, rax)?;
    }
    Ok(())
}

fn emit_bswap(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bits: u32,
) -> Result<()> {
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        asm.bswap(rax)?;
        if let Some(d) = dst {
            store64(asm, alloc, d, rax)?;
        }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        asm.bswap(eax)?;
        if let Some(d) = dst {
            store32(asm, alloc, d, eax)?;
        }
    }
    Ok(())
}

fn emit_load(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bytes: u32,
    use_fastmem: bool,
) -> Result<()> {
    load64(asm, alloc, a.args[0], SCRATCH1)?;

    if use_fastmem {
        let mut lbl_slow = asm.create_label();
        let mut lbl_done = asm.create_label();
        emit_fastmem_bounds_check(asm, bytes, lbl_slow)?;
        asm.mov(
            SCRATCH0,
            qword_ptr(CTX_REG + cpu_offsets::mem_base() as i32),
        )?;
        match bytes {
            1 => asm.movzx(eax, byte_ptr(SCRATCH0 + SCRATCH2))?,
            2 => asm.movzx(eax, word_ptr(SCRATCH0 + SCRATCH2))?,
            4 => asm.mov(eax, dword_ptr(SCRATCH0 + SCRATCH2))?,
            8 => asm.mov(rax, qword_ptr(SCRATCH0 + SCRATCH2))?,
            16 => asm.movdqu(xmm0, xmmword_ptr(SCRATCH0 + SCRATCH2))?,
            _ => return Err(Error::Backend("unsupported load width".into())),
        }
        asm.jmp(lbl_done)?;
        asm.set_label(&mut lbl_slow)?;
        emit_load_slow_path(asm, bytes, a.pc)?;
        asm.set_label(&mut lbl_done)?;
    } else {
        emit_load_slow_path(asm, bytes, a.pc)?;
    }

    if let Some(d) = dst {
        match bytes {
            1 | 2 | 4 => store32(asm, alloc, d, eax)?,
            8 => store64(asm, alloc, d, rax)?,
            16 => store_xmm_q(asm, alloc, d, xmm0)?,
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn emit_store(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    bytes: u32,
    use_fastmem: bool,
) -> Result<()> {
    match bytes {
        16 => load_xmm_q(asm, alloc, a.args[1], xmm0)?,
        8 => load64(asm, alloc, a.args[1], ARG3_REG)?,
        _ => load32(asm, alloc, a.args[1], gpr32(arg3_reg_id()))?,
    }
    load64(asm, alloc, a.args[0], SCRATCH1)?;

    if use_fastmem {
        let mut lbl_slow = asm.create_label();
        let mut lbl_done = asm.create_label();
        emit_fastmem_bounds_check(asm, bytes, lbl_slow)?;
        asm.mov(
            SCRATCH0,
            qword_ptr(CTX_REG + cpu_offsets::mem_base() as i32),
        )?;
        match bytes {
            1 => asm.mov(byte_ptr(SCRATCH0 + SCRATCH2), gpr8(arg3_reg_id()))?,
            2 => asm.mov(word_ptr(SCRATCH0 + SCRATCH2), gpr16(arg3_reg_id()))?,
            4 => asm.mov(dword_ptr(SCRATCH0 + SCRATCH2), gpr32(arg3_reg_id()))?,
            8 => asm.mov(qword_ptr(SCRATCH0 + SCRATCH2), ARG3_REG)?,
            16 => asm.movdqu(xmmword_ptr(SCRATCH0 + SCRATCH2), xmm0)?,
            _ => return Err(Error::Backend("unsupported store width".into())),
        }
        asm.jmp(lbl_done)?;
        asm.set_label(&mut lbl_slow)?;
        emit_store_slow_path(asm, bytes, a.pc)?;
        asm.set_label(&mut lbl_done)?;
    } else {
        emit_store_slow_path(asm, bytes, a.pc)?;
    }
    Ok(())
}

fn emit_fastmem_bounds_check(
    asm: &mut CodeAssembler,
    bytes: u32,
    lbl_slow: CodeLabel,
) -> Result<()> {
    // Check the guest address before subtracting the mapping base.  The old
    // `offset + bytes <= size` check allowed addresses below `mem_base_va` to
    // wrap into the mapping and access memory before the backing allocation.
    asm.mov(SCRATCH2, SCRATCH1)?;
    asm.cmp(
        SCRATCH2,
        qword_ptr(CTX_REG + cpu_offsets::mem_base_va() as i32),
    )?;
    asm.jb(lbl_slow)?;
    asm.sub(
        SCRATCH2,
        qword_ptr(CTX_REG + cpu_offsets::mem_base_va() as i32),
    )?;
    // Avoid offset+bytes overflow by comparing against size-bytes.
    asm.mov(
        SCRATCH0,
        qword_ptr(CTX_REG + cpu_offsets::mem_size() as i32),
    )?;
    asm.cmp(SCRATCH0, bytes as i32)?;
    asm.jb(lbl_slow)?;
    asm.sub(SCRATCH0, bytes as i32)?;
    asm.cmp(SCRATCH2, SCRATCH0)?;
    asm.ja(lbl_slow)?;
    Ok(())
}

fn emit_load_slow_path(asm: &mut CodeAssembler, bytes: u32, guest_pc: u64) -> Result<()> {
    if !matches!(bytes, 1 | 2 | 4 | 8 | 16) {
        return Err(Error::Backend("unsupported load width".into()));
    }
    asm.mov(SCRATCH0, guest_pc as i64)?;
    asm.mov(qword_ptr(CTX_REG + cpu_offsets::pc() as i32), SCRATCH0)?;
    asm.mov(SCRATCH3, bytes as i64)?;
    asm.mov(ARG0_REG, CTX_REG)?;
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.call(qword_ptr(CTX_REG + cpu_offsets::mem_read() as i32))?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    let io_off = cpu_offsets::io_value() as i32;
    match bytes {
        1 => asm.movzx(eax, byte_ptr(CTX_REG + io_off))?,
        2 => asm.movzx(eax, word_ptr(CTX_REG + io_off))?,
        4 => asm.mov(eax, dword_ptr(CTX_REG + io_off))?,
        8 => asm.mov(rax, qword_ptr(CTX_REG + io_off))?,
        16 => asm.movdqu(xmm0, xmmword_ptr(CTX_REG + io_off))?,
        _ => unreachable!(),
    }
    Ok(())
}

fn emit_store_slow_path(asm: &mut CodeAssembler, bytes: u32, guest_pc: u64) -> Result<()> {
    if !matches!(bytes, 1 | 2 | 4 | 8 | 16) {
        return Err(Error::Backend("unsupported store width".into()));
    }
    let io_off = cpu_offsets::io_value() as i32;
    asm.mov(SCRATCH0, guest_pc as i64)?;
    asm.mov(qword_ptr(CTX_REG + cpu_offsets::pc() as i32), SCRATCH0)?;
    match bytes {
        1 => asm.mov(byte_ptr(CTX_REG + io_off), gpr8(arg3_reg_id()))?,
        2 => asm.mov(word_ptr(CTX_REG + io_off), gpr16(arg3_reg_id()))?,
        4 => asm.mov(dword_ptr(CTX_REG + io_off), gpr32(arg3_reg_id()))?,
        8 => asm.mov(qword_ptr(CTX_REG + io_off), ARG3_REG)?,
        16 => asm.movdqu(xmmword_ptr(CTX_REG + io_off), xmm0)?,
        _ => unreachable!(),
    }
    asm.mov(SCRATCH3, bytes as i64)?;
    asm.mov(ARG0_REG, CTX_REG)?;
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.call(qword_ptr(CTX_REG + cpu_offsets::mem_write() as i32))?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    Ok(())
}

fn emit_load_ex(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bytes: u32,
) -> Result<()> {
    if !matches!(bytes, 1 | 2 | 4 | 8) {
        return Err(Error::Backend("unsupported ldex width".into()));
    }
    load64(asm, alloc, a.args[0], SCRATCH1)?;
    asm.mov(SCRATCH0, a.pc as i64)?;
    asm.mov(qword_ptr(CTX_REG + cpu_offsets::pc() as i32), SCRATCH0)?;
    asm.mov(SCRATCH3, bytes as i64)?;
    asm.mov(ARG0_REG, CTX_REG)?;
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.call(qword_ptr(CTX_REG + cpu_offsets::mem_read() as i32))?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    let io_off = cpu_offsets::io_value() as i32;
    match bytes {
        1 => asm.movzx(eax, byte_ptr(CTX_REG + io_off))?,
        2 => asm.movzx(eax, word_ptr(CTX_REG + io_off))?,
        4 => asm.mov(eax, dword_ptr(CTX_REG + io_off))?,
        8 => asm.mov(rax, qword_ptr(CTX_REG + io_off))?,
        _ => unreachable!(),
    }
    // Save the actual zero-extended loaded value before storing its SSA result
    // or reloading the address into scratch registers. STXR's callback compares
    // memory atomically with these bits, not just the local address/size pair.
    asm.mov(qword_ptr(CTX_REG + cpu_offsets::exclusive_value() as i32), rax)?;
    if let Some(d) = dst {
        match bytes {
            1 | 2 | 4 => store32(asm, alloc, d, eax)?,
            8 => store64(asm, alloc, d, rax)?,
            _ => unreachable!(),
        }
    }
    load64(asm, alloc, a.args[0], SCRATCH1)?;
    asm.mov(
        qword_ptr(CTX_REG + cpu_offsets::exclusive_addr() as i32),
        SCRATCH1,
    )?;
    asm.mov(
        byte_ptr(CTX_REG + cpu_offsets::exclusive_size() as i32),
        bytes as i32,
    )?;
    // A faulting exclusive load must not leave a usable local reservation.
    let mut load_succeeded = asm.create_label();
    asm.cmp(byte_ptr(CTX_REG + cpu_offsets::mem_fault() as i32), 0)?;
    asm.je(load_succeeded)?;
    asm.mov(byte_ptr(CTX_REG + cpu_offsets::exclusive_size() as i32), 0)?;
    asm.set_label(&mut load_succeeded)?;
    Ok(())
}

fn emit_store_ex(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bytes: u32,
) -> Result<()> {
    let mut lbl_fail = asm.create_label();
    let mut lbl_done = asm.create_label();

    load64(asm, alloc, a.args[0], SCRATCH1)?;
    asm.cmp(
        qword_ptr(CTX_REG + cpu_offsets::exclusive_addr() as i32),
        SCRATCH1,
    )?;
    asm.jne(lbl_fail)?;
    asm.cmp(
        byte_ptr(CTX_REG + cpu_offsets::exclusive_size() as i32),
        bytes as i32,
    )?;
    asm.jne(lbl_fail)?;

    if !matches!(bytes, 1 | 2 | 4 | 8) {
        return Err(Error::Backend("unsupported stex width".into()));
    }
    if bytes == 8 {
        load64(asm, alloc, a.args[1], ARG3_REG)?;
    } else {
        load32(asm, alloc, a.args[1], gpr32(arg3_reg_id()))?;
    }
    let io_off = cpu_offsets::io_value() as i32;
    asm.mov(SCRATCH0, a.pc as i64)?;
    asm.mov(qword_ptr(CTX_REG + cpu_offsets::pc() as i32), SCRATCH0)?;
    match bytes {
        1 => asm.mov(byte_ptr(CTX_REG + io_off), gpr8(arg3_reg_id()))?,
        2 => asm.mov(word_ptr(CTX_REG + io_off), gpr16(arg3_reg_id()))?,
        4 => asm.mov(dword_ptr(CTX_REG + io_off), gpr32(arg3_reg_id()))?,
        8 => asm.mov(qword_ptr(CTX_REG + io_off), ARG3_REG)?,
        _ => unreachable!(),
    }
    asm.mov(SCRATCH3, bytes as i64)?;
    asm.mov(ARG0_REG, CTX_REG)?;
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.call(qword_ptr(CTX_REG + cpu_offsets::mem_write_exclusive() as i32))?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    asm.jmp(lbl_done)?;

    asm.set_label(&mut lbl_fail)?;
    asm.mov(eax, 1i32)?;

    asm.set_label(&mut lbl_done)?;
    asm.mov(
        byte_ptr(CTX_REG + cpu_offsets::exclusive_size() as i32),
        0i32,
    )?;
    if let Some(d) = dst {
        store32(asm, alloc, d, eax)?;
    }
    Ok(())
}

fn emit_csel(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
) -> Result<()> {
    let cond = Cond::from_bits(a.imm as u8);
    let is_64 = matches!(a.op, Op::Csel64);

    load32(asm, alloc, a.args[2], edx)?;
    emit_cond_check_byte(asm, cond)?;
    asm.test(al, al)?;
    if is_64 {
        load64(asm, alloc, a.args[1], SCRATCH1)?;
        load64(asm, alloc, a.args[0], SCRATCH2)?;
        asm.cmovne(SCRATCH1, SCRATCH2)?;
        if let Some(d) = dst {
            store64(asm, alloc, d, SCRATCH1)?;
        }
    } else {
        load32(asm, alloc, a.args[1], eax)?;
        load32(asm, alloc, a.args[0], gpr32(scratch1_id()))?;
        asm.cmovne(eax, gpr32(scratch1_id()))?;
        if let Some(d) = dst {
            store32(asm, alloc, d, eax)?;
        }
    }
    Ok(())
}

pub fn emit_cond_check_byte(asm: &mut CodeAssembler, cond: Cond) -> Result<()> {
    let tt = crate::arch::COND_TRUTH[cond as usize] as i32;
    asm.mov(eax, tt)?;
    asm.bt(eax, edx)?;
    asm.setc(al)?;
    Ok(())
}

#[inline]
fn scratch1_id() -> u8 {
    #[cfg(target_os = "windows")]
    {
        2
    }
    #[cfg(not(target_os = "windows"))]
    {
        6
    }
}

#[inline]
#[allow(dead_code)]
fn scratch3_id() -> u8 {
    #[cfg(target_os = "windows")]
    {
        8
    }
    #[cfg(not(target_os = "windows"))]
    {
        2
    }
}

#[inline]
fn arg3_reg_id() -> u8 {
    #[cfg(target_os = "windows")]
    {
        9
    }
    #[cfg(not(target_os = "windows"))]
    {
        1
    }
}

#[derive(Clone, Copy)]
enum FpBinKind {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

fn emit_fbinop(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    k: FpBinKind,
    bits: u32,
) -> Result<()> {
    if bits == 64 {
        load_xmm_d(asm, alloc, a.args[0], xmm0)?;
        load_xmm_d(asm, alloc, a.args[1], xmm1)?;
        match k {
            FpBinKind::Add => asm.addsd(xmm0, xmm1)?,
            FpBinKind::Sub => asm.subsd(xmm0, xmm1)?,
            FpBinKind::Mul => asm.mulsd(xmm0, xmm1)?,
            FpBinKind::Div => asm.divsd(xmm0, xmm1)?,
            FpBinKind::Max => asm.maxsd(xmm0, xmm1)?,
            FpBinKind::Min => asm.minsd(xmm0, xmm1)?,
        }
        if let Some(d) = dst {
            store_xmm_d(asm, alloc, d, xmm0)?;
        }
    } else {
        load_xmm_s(asm, alloc, a.args[0], xmm0)?;
        load_xmm_s(asm, alloc, a.args[1], xmm1)?;
        match k {
            FpBinKind::Add => asm.addss(xmm0, xmm1)?,
            FpBinKind::Sub => asm.subss(xmm0, xmm1)?,
            FpBinKind::Mul => asm.mulss(xmm0, xmm1)?,
            FpBinKind::Div => asm.divss(xmm0, xmm1)?,
            FpBinKind::Max => asm.maxss(xmm0, xmm1)?,
            FpBinKind::Min => asm.minss(xmm0, xmm1)?,
        }
        if let Some(d) = dst {
            store_xmm_s(asm, alloc, d, xmm0)?;
        }
    }
    Ok(())
}

fn emit_fcvt_precision(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    src_is_double: bool,
) -> Result<()> {
    if src_is_double {
        load_xmm_d(asm, alloc, a.args[0], xmm0)?;
        asm.cvtsd2ss(xmm0, xmm0)?;
        if let Some(d) = dst {
            store_xmm_s(asm, alloc, d, xmm0)?;
        }
    } else {
        load_xmm_s(asm, alloc, a.args[0], xmm0)?;
        asm.cvtss2sd(xmm0, xmm0)?;
        if let Some(d) = dst {
            store_xmm_d(asm, alloc, d, xmm0)?;
        }
    }
    Ok(())
}

fn emit_fcvt_zs(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    src_is_double: bool,
    dst_is_x: bool,
) -> Result<()> {
    if src_is_double {
        load_xmm_d(asm, alloc, a.args[0], xmm0)?;
        if dst_is_x {
            asm.cvttsd2si(rax, xmm0)?;
        } else {
            asm.cvttsd2si(eax, xmm0)?;
        }
    } else {
        load_xmm_s(asm, alloc, a.args[0], xmm0)?;
        if dst_is_x {
            asm.cvttss2si(rax, xmm0)?;
        } else {
            asm.cvttss2si(eax, xmm0)?;
        }
    }

    // x86 returns the signed minimum for every invalid conversion. Arm
    // saturates overflow instead and converts NaN to zero. Ordinary results
    // retain the native conversion path; classify only its ambiguous sentinel.
    let mut done = asm.create_label();
    let mut nan = asm.create_label();
    if dst_is_x {
        asm.mov(SCRATCH1, i64::MIN)?;
        asm.cmp(rax, SCRATCH1)?;
    } else {
        asm.cmp(eax, i32::MIN)?;
    }
    asm.jne(done)?;
    asm.pxor(xmm1, xmm1)?;
    if src_is_double {
        asm.ucomisd(xmm0, xmm1)?;
    } else {
        asm.ucomiss(xmm0, xmm1)?;
    }
    asm.jp(nan)?;
    // Negative overflow and a valid signed-minimum result both stay minimum.
    asm.jbe(done)?;
    if dst_is_x {
        asm.mov(rax, i64::MAX)?;
    } else {
        asm.mov(eax, i32::MAX)?;
    }
    asm.jmp(done)?;
    asm.set_label(&mut nan)?;
    asm.xor(eax, eax)?;
    asm.set_label(&mut done)?;

    if let Some(d) = dst {
        if dst_is_x {
            store64(asm, alloc, d, rax)?;
        } else {
            store32(asm, alloc, d, eax)?;
        }
    }
    Ok(())
}

// Process special values in architectural addend/multiplicand/multiplier
// order. Signaling NaNs take precedence over quiet NaNs; generated invalid
// operations produce the positive default NaN. FPSR updates remain separate.
fn scalar_fma_special(n: u64, m: u64, a: u64, double: bool) -> Option<u64> {
    let (exponent, fraction, quiet, sign) = if double {
        (0x7ff0_0000_0000_0000, 0x000f_ffff_ffff_ffff,
            0x0008_0000_0000_0000, 0x8000_0000_0000_0000)
    } else {
        (0x7f80_0000, 0x007f_ffff, 0x0040_0000, 0x8000_0000)
    };
    let nan = |v: u64| v & exponent == exponent && v & fraction != 0;
    let infinity = |v: u64| v & !sign == exponent;
    let zero = |v: u64| v & !sign == 0;
    let invalid_product = (infinity(n) && zero(m)) || (zero(n) && infinity(m));
    for operand in [a, n, m] {
        if nan(operand) && operand & quiet == 0 {
            return Some(operand | quiet);
        }
    }
    // 0*Inf overrides a quiet NaN addend (but not signaling NaNs).
    if nan(a) && invalid_product {
        return Some(exponent | quiet);
    }
    for operand in [a, n, m] {
        if nan(operand) {
            return Some(operand | quiet);
        }
    }
    let invalid_sum = infinity(a) && (infinity(n) || infinity(m))
        && ((n ^ m ^ a) & sign != 0);
    if invalid_product || invalid_sum {
        return Some(exponent | quiet);
    }
    None
}

extern "C" fn scalar_fma_s(n: u64, m: u64, a: u64) -> u64 {
    let (n, m, a) = (n as u32 as u64, m as u32 as u64, a as u32 as u64);
    if let Some(value) = scalar_fma_special(n, m, a, false) {
        return value;
    }
    // Rust mul_add specifies one correctly rounded fused operation, including
    // hosts without FMA. Multiplying in f64 and narrowing is not equivalent.
    f32::from_bits(n as u32).mul_add(f32::from_bits(m as u32), f32::from_bits(a as u32)).to_bits() as u64
}

extern "C" fn scalar_fma_d(n: u64, m: u64, a: u64) -> u64 {
    if let Some(value) = scalar_fma_special(n, m, a, true) {
        return value;
    }
    f64::from_bits(n).mul_add(f64::from_bits(m), f64::from_bits(a)).to_bits()
}

fn emit_op_scalar_fma(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let double = a.op == Op::Fmadd64;
    #[cfg(target_os = "windows")]
    let arguments = [rcx, rdx, r8];
    #[cfg(not(target_os = "windows"))]
    let arguments = [rdi, rsi, rdx];
    for (argument, value) in arguments.into_iter().zip(a.args) {
        if double {
            load64(asm, alloc, value, argument)?;
        } else {
            let register = match argument {
                r if r == rcx => ecx,
                r if r == rdx => edx,
                r if r == r8 => r8d,
                r if r == rdi => edi,
                _ => esi,
            };
            load32(asm, alloc, value, register)?;
        }
    }
    let features = crate::backend::cpu_features::active_features();
    let hardware = features.has_avx && features.has_fma;
    let mut helper = asm.create_label();
    let mut done = asm.create_label();
    if hardware {
        if double {
            asm.movq(xmm1, arguments[0])?;
            asm.movq(xmm2, arguments[1])?;
            asm.movq(xmm0, arguments[2])?;
            asm.vfmadd231sd(xmm0, xmm1, xmm2)?;
            asm.ucomisd(xmm0, xmm0)?;
        } else {
            // MOVQ is also safe for S: arguments were zero extended above.
            asm.movq(xmm1, arguments[0])?;
            asm.movq(xmm2, arguments[1])?;
            asm.movq(xmm0, arguments[2])?;
            asm.vfmadd231ss(xmm0, xmm1, xmm2)?;
            asm.ucomiss(xmm0, xmm0)?;
        }
        // Native finite/infinite values are fused. NaN selection and invalid
        // default-NaN differ between x86 and Arm, so use the shared helper.
        asm.jp(helper)?;
        if double { asm.movq(rax, xmm0)?; } else { asm.movd(eax, xmm0)?; }
        asm.jmp(done)?;
    }
    asm.set_label(&mut helper)?;
    if hardware { asm.vzeroupper()?; }
    let function = if double { scalar_fma_d as *const () } else { scalar_fma_s as *const () };
    asm.mov(rax, function as usize as i64)?;
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.call(rax)?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    asm.set_label(&mut done)?;
    if let Some(destination) = dst_of(&a, idx) {
        if double { store64(asm, alloc, destination, rax)?; }
        else { store32(asm, alloc, destination, eax)?; }
    }
    Ok(())
}

fn emit_op_reciprocal(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let double = matches!(a.op, Op::FrsqrtEstimate64 | Op::FrsqrtStep64 | Op::FrecipEstimate64 | Op::FrecipStep64);
    let step = matches!(a.op, Op::FrsqrtStep32 | Op::FrsqrtStep64 | Op::FrecipStep32 | Op::FrecipStep64);
    let reciprocal = matches!(a.op, Op::FrecipEstimate32 | Op::FrecipEstimate64 | Op::FrecipStep32 | Op::FrecipStep64);
    // Context first, then one/two integer bit-pattern arguments. These ops
    // declare all caller-saved GPR/XMM clobbers, just like the FMA helper.
    for (argument, value) in [SCRATCH1, SCRATCH3].into_iter().zip(a.args).take(if step { 2 } else { 1 }) {
        if double { load64(asm, alloc, value, argument)?; }
        else {
            let register = match argument {
                r if r == rdx => edx,
                r if r == r8 => r8d,
                r if r == rsi => esi,
                _ => edx,
            };
            load32(asm, alloc, value, register)?;
        }
    }
    asm.mov(ARG0_REG, CTX_REG)?;
    let function = match (step, double, reciprocal) {
        (false, false, false) => crate::backend::reciprocal::estimate_s as *const (),
        (false, true, false) => crate::backend::reciprocal::estimate_d as *const (),
        (true, false, false) => crate::backend::reciprocal::step_s as *const (),
        (true, true, false) => crate::backend::reciprocal::step_d as *const (),
        (false, false, true) => crate::backend::reciprocal::reciprocal_estimate_s as *const (),
        (false, true, true) => crate::backend::reciprocal::reciprocal_estimate_d as *const (),
        (true, false, true) => crate::backend::reciprocal::reciprocal_step_s as *const (),
        (true, true, true) => crate::backend::reciprocal::reciprocal_step_d as *const (),
    };
    asm.mov(rax, function as usize as i64)?;
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.call(rax)?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    if let Some(destination) = dst_of(&a, idx) {
        if double { store64(asm, alloc, destination, rax)?; }
        else { store32(asm, alloc, destination, eax)?; }
    }
    Ok(())
}

fn emit_op_absolute_difference(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let double = a.op == Op::FabsDiff64;
    // Preserve original N/M bit patterns in integer ABI argument registers
    // before SSE subtraction, whose mixed-NaN priority differs from Arm.
    let narrow_register = |register| match register {
        r if r == rdx => edx,
        r if r == r8 => r8d,
        r if r == rsi => esi,
        _ => unreachable!(),
    };
    for (argument, value) in [SCRATCH1, SCRATCH3].into_iter().zip(a.args).take(2) {
        if double { load64(asm, alloc, value, argument)?; }
        else { load32(asm, alloc, value, narrow_register(argument))?; }
    }
    let mut helper = asm.create_label();
    let mut done = asm.create_label();
    if double {
        asm.movq(xmm0, SCRATCH1)?;
        asm.movq(xmm1, SCRATCH3)?;
        asm.subsd(xmm0, xmm1)?;
        asm.ucomisd(xmm0, xmm0)?;
    } else {
        asm.movd(xmm0, narrow_register(SCRATCH1))?;
        asm.movd(xmm1, narrow_register(SCRATCH3))?;
        asm.subss(xmm0, xmm1)?;
        asm.ucomiss(xmm0, xmm0)?;
    }
    asm.jp(helper)?;
    // The common numeric path is native target-width FSUB then exact sign
    // clearing, without a helper call or any widened subtraction.
    if double {
        asm.movq(rax, xmm0)?;
        asm.btr(rax, 63)?;
    } else {
        asm.movd(eax, xmm0)?;
        asm.and(eax, 0x7fff_ffff)?;
    }
    asm.jmp(done)?;
    asm.set_label(&mut helper)?;
    asm.mov(ARG0_REG, CTX_REG)?;
    let function = if double {
        crate::backend::absolute_difference::double as *const ()
    } else {
        crate::backend::absolute_difference::single as *const ()
    };
    asm.mov(rax, function as usize as i64)?;
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.call(rax)?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    asm.set_label(&mut done)?;
    if let Some(destination) = dst_of(&a, idx) {
        if double { store64(asm, alloc, destination, rax)?; }
        else { store32(asm, alloc, destination, eax)?; }
    }
    Ok(())
}

fn emit_op_fp_minmax_number(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let double = a.op == Op::FpMinMaxNum64;
    if double {
        load64(asm, alloc, a.args[0], SCRATCH1)?;
        load64(asm, alloc, a.args[1], SCRATCH3)?;
    } else {
        // Narrow lanes must also clear unused spill/XMM bits before crossing
        // the integer ABI, just as a guest W register write zero-extends.
        #[cfg(target_os = "windows")]
        let (n_reg, m_reg) = (edx, r8d);
        #[cfg(not(target_os = "windows"))]
        let (n_reg, m_reg) = (esi, edx);
        load32(asm, alloc, a.args[0], n_reg)?;
        load32(asm, alloc, a.args[1], m_reg)?;
    }
    asm.mov(ARG0_REG, CTX_REG)?;
    asm.mov(ARG3_REG, a.imm as i64)?;
    let function = if double {
        crate::backend::fp_minmax_number::double as *const ()
    } else {
        crate::backend::fp_minmax_number::single as *const ()
    };
    asm.mov(rax, function as usize as i64)?;
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.call(rax)?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    if let Some(destination) = dst_of(&a, idx) {
        if double { store64(asm, alloc, destination, rax)?; }
        else { store32(asm, alloc, destination, eax)?; }
    }
    Ok(())
}

fn emit_op_integer_reduce(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    load64(asm, alloc, a.args[0], ARG0_REG)?;
    load64(asm, alloc, a.args[1], SCRATCH1)?;
    asm.mov(SCRATCH3, a.imm as i64)?;
    asm.mov(rax, crate::backend::integer_minmax_reduce::integer_minmax_reduce as *const () as usize as i64)?;
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.call(rax)?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    if let Some(destination) = dst_of(&a, idx) { store32(asm, alloc, destination, eax)?; }
    Ok(())
}

fn emit_op_fp_reduce(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    load64(asm, alloc, a.args[0], SCRATCH1)?;
    load64(asm, alloc, a.args[1], SCRATCH3)?;
    asm.mov(ARG0_REG, CTX_REG)?;
    asm.mov(ARG3_REG, a.imm as i64)?;
    asm.mov(rax, crate::backend::fp_reduce::reduce as *const () as usize as i64)?;
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.call(rax)?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    if let Some(destination) = dst_of(&a, idx) { store32(asm, alloc, destination, eax)?; }
    Ok(())
}

fn emit_op_saturating_narrow(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    load64(asm, alloc, a.args[0], SCRATCH1)?;
    load64(asm, alloc, a.args[1], SCRATCH3)?;
    asm.mov(ARG0_REG, CTX_REG)?;
    asm.mov(ARG3_REG, a.imm as i64)?;
    asm.mov(rax, crate::backend::saturating::narrow as *const () as usize as i64)?;
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.call(rax)?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    if let Some(destination) = dst_of(&a, idx) { store64(asm, alloc, destination, rax)?; }
    Ok(())
}

fn emit_op_fcvt_zu(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let double = matches!(a.op, Op::FcvtZuDW | Op::FcvtZuDX);
    let dst_is_x = matches!(a.op, Op::FcvtZuSX | Op::FcvtZuDX);
    let mut zero = asm.create_label();
    let mut maximum = asm.create_label();
    let mut low = asm.create_label();
    let mut done = asm.create_label();
    if double {
        load_xmm_d(asm, alloc, a.args[0], xmm0)?;
    } else {
        load_xmm_s(asm, alloc, a.args[0], xmm0)?;
    }
    asm.pxor(xmm1, xmm1)?;
    if double {
        asm.ucomisd(xmm0, xmm1)?;
    } else {
        asm.ucomiss(xmm0, xmm1)?;
    }
    // NaN, negative values and both signed zeros all convert to zero.
    asm.jp(zero)?;
    asm.jbe(zero)?;
    let width = if dst_is_x { 64 } else { 32 };
    if double {
        asm.mov(rcx, ((1023 + width as u64) << 52) as i64)?;
        asm.movq(xmm1, rcx)?;
        asm.ucomisd(xmm0, xmm1)?;
    } else {
        asm.mov(ecx, ((127 + width) << 23) as i32)?;
        asm.movd(xmm1, ecx)?;
        asm.ucomiss(xmm0, xmm1)?;
    }
    asm.jae(maximum)?;
    if dst_is_x {
        if double {
            asm.mov(rcx, ((1023u64 + 63) << 52) as i64)?;
            asm.movq(xmm1, rcx)?;
            asm.ucomisd(xmm0, xmm1)?;
        } else {
            asm.mov(ecx, ((127 + 63) << 23) as i32)?;
            asm.movd(xmm1, ecx)?;
            asm.ucomiss(xmm0, xmm1)?;
        }
        asm.jb(low)?;
        // In [2^63,2^64), subtracting 2^63 is exact (Sterbenz's lemma)
        // and leaves a representable signed integer. Restore its high bit.
        if double {
            asm.subsd(xmm0, xmm1)?;
            asm.cvttsd2si(rax, xmm0)?;
        } else {
            asm.subss(xmm0, xmm1)?;
            asm.cvttss2si(rax, xmm0)?;
        }
        asm.mov(rcx, i64::MIN)?;
        asm.or(rax, rcx)?;
        asm.jmp(done)?;
    }
    asm.set_label(&mut low)?;
    // The complete unsigned W range fits a signed X conversion as well.
    if double {
        asm.cvttsd2si(rax, xmm0)?;
    } else {
        asm.cvttss2si(rax, xmm0)?;
    }
    asm.jmp(done)?;
    asm.set_label(&mut maximum)?;
    if dst_is_x {
        asm.mov(rax, -1i64)?;
    } else {
        asm.mov(eax, -1i32)?;
    }
    asm.jmp(done)?;
    asm.set_label(&mut zero)?;
    asm.xor(eax, eax)?;
    asm.set_label(&mut done)?;
    if let Some(dst) = dst_of(&a, idx) {
        if dst_is_x {
            store64(asm, alloc, dst, rax)?;
        } else {
            store32(asm, alloc, dst, eax)?;
        }
    }
    Ok(())
}

fn emit_op_crc32(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let polynomial = if a.op.base() == Op::Crc32Ieee8.base() {
        0xEDB8_8320u32
    } else {
        0x82F6_3B78u32
    };
    load32(asm, alloc, a.args[0], eax)?;
    load64(asm, alloc, a.args[1], rcx)?;
    // Both architectural polynomials use reflected bit order, starting with
    // the least significant input byte. x86 CRC32 only supports Castagnoli;
    // emit portable integer code so IEEE and SSE4.1 hosts are correct too.
    for byte in 0..a.op.size_bytes() {
        asm.xor(al, cl)?;
        for _ in 0..8 {
            asm.shr(eax, 1)?;
            asm.sbb(edx, edx)?; // -old_crc_bit_0, using SHR's carry.
            asm.and(edx, polynomial as i32)?;
            asm.xor(eax, edx)?;
        }
        if byte + 1 < a.op.size_bytes() {
            asm.shr(rcx, 8)?;
        }
    }
    if let Some(dst) = dst_of(&a, idx) {
        store32(asm, alloc, dst, eax)?;
    }
    Ok(())
}

fn emit_scvtf(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    dst_is_double: bool,
    src_is_x: bool,
) -> Result<()> {
    if src_is_x {
        load64(asm, alloc, a.args[0], rax)?;
        if dst_is_double {
            asm.cvtsi2sd(xmm0, rax)?;
        } else {
            asm.cvtsi2ss(xmm0, rax)?;
        }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        if dst_is_double {
            asm.cvtsi2sd(xmm0, eax)?;
        } else {
            asm.cvtsi2ss(xmm0, eax)?;
        }
    }
    if let Some(d) = dst {
        if dst_is_double {
            store_xmm_d(asm, alloc, d, xmm0)?;
        } else {
            store_xmm_s(asm, alloc, d, xmm0)?;
        }
    }
    Ok(())
}

fn emit_op_ucvtf(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    let src_is_x = matches!(a.op, Op::UcvtfXS | Op::UcvtfXD);
    let double = matches!(a.op, Op::UcvtfWD | Op::UcvtfXD);
    let mut direct = asm.create_label();
    let mut done = asm.create_label();
    if src_is_x {
        load64(asm, alloc, a.args[0], rax)?;
        asm.test(rax, rax)?;
        asm.jns(direct)?;
        // Unsigned values >= 2^63 do not fit the signed SSE conversion.
        // Sticky-half retains discarded low bits for nearest-even rounding;
        // converting that positive half and doubling is correctly rounded.
        asm.mov(rdx, rax)?;
        asm.and(edx, 1)?;
        asm.shr(rax, 1)?;
        asm.or(rax, rdx)?;
        if double {
            asm.cvtsi2sd(xmm0, rax)?;
            asm.addsd(xmm0, xmm0)?;
        } else {
            asm.cvtsi2ss(xmm0, rax)?;
            asm.addss(xmm0, xmm0)?;
        }
        asm.jmp(done)?;
    } else {
        // Zero extension puts the entire unsigned W range within signed X.
        load32(asm, alloc, a.args[0], eax)?;
    }
    asm.set_label(&mut direct)?;
    if double {
        asm.cvtsi2sd(xmm0, rax)?;
    } else {
        asm.cvtsi2ss(xmm0, rax)?;
    }
    asm.set_label(&mut done)?;
    if let Some(dst) = dst_of(&a, idx) {
        if double {
            store_xmm_d(asm, alloc, dst, xmm0)?;
        } else {
            store_xmm_s(asm, alloc, dst, xmm0)?;
        }
    }
    Ok(())
}

fn emit_fneg(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bits: u32,
) -> Result<()> {
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        asm.mov(rcx, 0x8000_0000_0000_0000_u64 as i64)?;
        asm.xor(rax, rcx)?;
        if let Some(d) = dst {
            store64(asm, alloc, d, rax)?;
        }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        asm.xor(eax, 0x8000_0000_u32 as i32)?;
        if let Some(d) = dst {
            store32(asm, alloc, d, eax)?;
        }
    }
    Ok(())
}

fn emit_fabs(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bits: u32,
) -> Result<()> {
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        asm.mov(rcx, 0x7FFF_FFFF_FFFF_FFFF_u64 as i64)?;
        asm.and(rax, rcx)?;
        if let Some(d) = dst {
            store64(asm, alloc, d, rax)?;
        }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        asm.and(eax, 0x7FFF_FFFF_u32 as i32)?;
        if let Some(d) = dst {
            store32(asm, alloc, d, eax)?;
        }
    }
    Ok(())
}

fn emit_fsqrt(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bits: u32,
) -> Result<()> {
    if bits == 64 {
        load_xmm_d(asm, alloc, a.args[0], xmm0)?;
        asm.sqrtsd(xmm0, xmm0)?;
        if let Some(d) = dst {
            store_xmm_d(asm, alloc, d, xmm0)?;
        }
    } else {
        load_xmm_s(asm, alloc, a.args[0], xmm0)?;
        asm.sqrtss(xmm0, xmm0)?;
        if let Some(d) = dst {
            store_xmm_s(asm, alloc, d, xmm0)?;
        }
    }
    Ok(())
}

fn emit_fcmp(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, bits: u32) -> Result<()> {
    if bits == 64 {
        load_xmm_d(asm, alloc, a.args[0], xmm0)?;
        load_xmm_d(asm, alloc, a.args[1], xmm1)?;
        asm.ucomisd(xmm0, xmm1)?;
    } else {
        load_xmm_s(asm, alloc, a.args[0], xmm0)?;
        load_xmm_s(asm, alloc, a.args[1], xmm1)?;
        asm.ucomiss(xmm0, xmm1)?;
    }
    asm.setp(r8b)?;
    asm.setnp(cl)?;
    asm.setz(r9b)?;
    asm.setc(r10b)?;
    asm.and(r9b, cl)?;
    asm.and(r10b, cl)?;
    asm.mov(r11b, r10b)?;
    asm.xor(r11b, 1i32)?;
    asm.shl(r10b, 3i32)?;
    asm.shl(r9b, 2i32)?;
    asm.shl(r11b, 1i32)?;
    asm.or(r10b, r9b)?;
    asm.or(r10b, r11b)?;
    asm.or(r10b, r8b)?;
    asm.mov(byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32), r10b)?;
    Ok(())
}

fn emit_mrs(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
) -> Result<()> {
    use crate::arch::sysreg;
    let d = match dst {
        Some(d) => d,
        None => return Ok(()),
    };
    let id = a.imm as u16;
    match id {
        sysreg::TPIDR_EL0 => {
            asm.mov(
                SCRATCH0,
                qword_ptr(CTX_REG + cpu_offsets::tpidr_el0() as i32),
            )?;
        }
        sysreg::TPIDRRO_EL0 => {
            asm.mov(
                SCRATCH0,
                qword_ptr(CTX_REG + cpu_offsets::tpidrro_el0() as i32),
            )?;
        }
        sysreg::NZCV => {
            asm.movzx(eax, byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32))?;
            asm.shl(rax, 28i32)?;
        }
        sysreg::FPCR => {
            asm.mov(eax, dword_ptr(CTX_REG + cpu_offsets::fpcr() as i32))?;
        }
        sysreg::FPSR => {
            asm.mov(eax, dword_ptr(CTX_REG + cpu_offsets::fpsr() as i32))?;
        }
        sysreg::CTR_EL0 => {
            asm.mov(SCRATCH0, 0x8444_8004u64 as i64)?;
        }
        sysreg::DCZID_EL0 => {
            asm.mov(SCRATCH0, 0x4i64)?;
        }
        sysreg::MIDR_EL1 => {
            asm.mov(SCRATCH0, 0x412F_D050u64 as i64)?;
        }
        sysreg::MPIDR_EL1 => {
            asm.mov(SCRATCH0, qword_ptr(CTX_REG + cpu_offsets::core_id() as i32))?;
            asm.and(SCRATCH0, 0xffi32)?;
            asm.or(SCRATCH0, i32::MIN)?;
        }
        sysreg::CNTFRQ_EL0 => {
            asm.mov(
                SCRATCH0,
                qword_ptr(CTX_REG + cpu_offsets::cntfrq_el0() as i32),
            )?;
        }
        sysreg::CNTPCT_EL0 | sysreg::CNTVCT_EL0 => {
            asm.mov(ARG0_REG, CTX_REG)?;
            asm.sub(rsp, CALL_PRECALL_SUB)?;
            asm.call(qword_ptr(CTX_REG + cpu_offsets::read_cntpct() as i32))?;
            asm.add(rsp, CALL_PRECALL_SUB)?;
        }
        _ => {
            return Err(Error::Unsupported {
                pc: 0,
                opcode: ((Op::Mrs as u32) << 16) | id as u32,
            });
        }
    }
    store64(asm, alloc, d, SCRATCH0)?;
    Ok(())
}

fn emit_msr(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet) -> Result<()> {
    use crate::arch::sysreg;
    let id = a.imm as u16;
    load64(asm, alloc, a.args[0], SCRATCH0)?;
    match id {
        sysreg::TPIDR_EL0 => {
            asm.mov(
                qword_ptr(CTX_REG + cpu_offsets::tpidr_el0() as i32),
                SCRATCH0,
            )?;
        }
        sysreg::TPIDRRO_EL0 => {
            asm.mov(
                qword_ptr(CTX_REG + cpu_offsets::tpidrro_el0() as i32),
                SCRATCH0,
            )?;
        }
        sysreg::NZCV => {
            asm.shr(rax, 28i32)?;
            asm.and(eax, 0xFi32)?;
            asm.mov(byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32), al)?;
        }
        sysreg::FPCR => {
            asm.mov(dword_ptr(CTX_REG + cpu_offsets::fpcr() as i32), eax)?;
        }
        sysreg::FPSR => {
            asm.mov(dword_ptr(CTX_REG + cpu_offsets::fpsr() as i32), eax)?;
        }
        _ => {}
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum VecBinKind {
    Add(u32),
    Sub(u32),
    And,
    Orr,
    Eor,
    Bic,
    Orn,
}

fn emit_vec_binop(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    kind: VecBinKind,
) -> Result<()> {
    let d = dst.unwrap();
    let q_form = (a.imm & 1) != 0;

    let (working_src, other_src) = match kind {
        VecBinKind::Bic => (a.args[1], a.args[0]),
        _ => (a.args[0], a.args[1]),
    };

    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, working_src, working)?;
    let other = get_xmm_q(asm, alloc, other_src, xmm1)?;

    match kind {
        VecBinKind::Add(sz) => match sz {
            0 => asm.paddb(working, other)?,
            1 => asm.paddw(working, other)?,
            2 => asm.paddd(working, other)?,
            3 => asm.paddq(working, other)?,
            _ => unreachable!(),
        },
        VecBinKind::Sub(sz) => match sz {
            0 => asm.psubb(working, other)?,
            1 => asm.psubw(working, other)?,
            2 => asm.psubd(working, other)?,
            3 => asm.psubq(working, other)?,
            _ => unreachable!(),
        },
        VecBinKind::And => asm.pand(working, other)?,
        VecBinKind::Orr => asm.por(working, other)?,
        VecBinKind::Eor => asm.pxor(working, other)?,
        VecBinKind::Bic => asm.pandn(working, other)?,
        VecBinKind::Orn => {
            asm.pcmpeqd(xmm2, xmm2)?;
            asm.pxor(xmm2, other)?;
            asm.por(working, xmm2)?;
        }
    }

    if !q_form {
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)?;
    Ok(())
}
