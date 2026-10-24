use iced_x86::BlockEncoderOptions;
use iced_x86::code_asm::*;

use crate::arch::Cond;
use crate::backend::abi::CTX_REG;
use crate::backend::isel::emit_armlet;
use crate::backend::operand::load64;
use crate::backend::prologue::{emit_epilogue, emit_prologue};
use crate::backend::regalloc::{ALLOCATABLE_GPRS, compute_live_ranges_with, linear_scan};
use crate::backend::regalloc::{Allocation, Loc};
use crate::backend::thunk::ThunkKind;
use crate::error::{Error, Result};
use crate::ir::block::ExceptionKind;
use crate::ir::{Armlet, Block, Op, Terminal, ValueRef};
use crate::jit::context::cpu_offsets;

pub struct EmittedBlock {
    pub code: Vec<u8>,
    pub chains: Vec<ChainSite>,
    pub direct_sites: Vec<DirectSite>,
    pub body_offset: u32,
    pub thunk_kind: ThunkKind,
}

/// Do not use Allocation.used_xmms: it describes the final active/free state,
/// not every location written earlier in the block. A final machine-code scan
/// also rejects fixed-register/implicit vector writers missing from metadata.
fn emitted_thunk_kind(block: &Block, alloc: &Allocation, code: &[u8]) -> ThunkKind {
    #[cfg(target_os = "windows")]
    {
        use iced_x86::{Decoder, DecoderOptions, Mnemonic, OpKind, Register};
        if alloc.locs.iter().any(|loc| matches!(loc, Loc::Xmm(6..=15)))
            || block.iter_live().any(|(_, armlet)| {
                crate::backend::clobbers::clobbers_for_op(armlet.op).xmm.bits() & 0xffc0 != 0
            })
        {
            return ThunkKind::Full;
        }
        let mut decoder = Decoder::new(64, code, DecoderOptions::NONE);
        while decoder.can_decode() {
            let instruction = decoder.decode();
            if instruction.is_invalid()
                || matches!(instruction.mnemonic(), Mnemonic::Fxrstor | Mnemonic::Fxrstor64
                    | Mnemonic::Xrstor | Mnemonic::Xrstor64 | Mnemonic::Xrstors
                    | Mnemonic::Xrstors64 | Mnemonic::Vzeroall | Mnemonic::Vzeroupper)
            {
                return ThunkKind::Full;
            }
            for operand in 0..instruction.op_count() {
                if instruction.op_kind(operand) == OpKind::Register {
                    let register = instruction.op_register(operand);
                    if (Register::XMM6..=Register::XMM15).contains(&register)
                        || (Register::YMM6..=Register::YMM15).contains(&register)
                        || (Register::ZMM6..=Register::ZMM15).contains(&register)
                    {
                        return ThunkKind::Full;
                    }
                }
            }
            let memory_index = instruction.memory_index();
            if (Register::XMM6..=Register::XMM15).contains(&memory_index)
                || (Register::YMM6..=Register::YMM15).contains(&memory_index)
                || (Register::ZMM6..=Register::ZMM15).contains(&memory_index)
            {
                return ThunkKind::Full;
            }
        }
        if decoder.position() != code.len() {
            return ThunkKind::Full;
        }
        ThunkKind::XmmFree
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = (block, alloc, code);
        ThunkKind::Full
    }
}

#[cfg(all(test, target_os = "windows"))]
mod thunk_certificate_tests {
    use super::*;
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};

    fn certificate(locs: Vec<Loc>, code: &[u8]) -> ThunkKind {
        let allocation = Allocation { locs, spill_bytes: 0, used_xmms: 0 };
        emitted_thunk_kind(&Block::new(0x1000), &allocation, code)
    }

    #[test]
    fn expired_historical_xmm_locations_require_full_preservation() {
        for register in 6..=15 {
            assert_eq!(certificate(vec![Loc::Xmm(register)], &[0xc3]), ThunkKind::Full);
        }
        assert_eq!(certificate(vec![Loc::Reg(3)], &[0xc3]), ThunkKind::XmmFree);
    }

    #[test]
    fn fixed_vector_operands_and_malformed_code_require_full_preservation() {
        for code in [&[0xf3, 0x0f, 0x6f, 0xf0][..], &[0x66, 0x0f, 0xef, 0xff][..], &[0x0f][..]] {
            assert_eq!(certificate(Vec::new(), code), ThunkKind::Full);
        }
        assert_eq!(certificate(Vec::new(), &[0x48, 0x83, 0xc0, 0x01, 0xc3]), ThunkKind::XmmFree);
    }

    #[test]
    fn implicit_restore_and_zero_vector_writers_include_all_64_bit_aliases() {
        for (code, mnemonic) in [
            (&[0x0f, 0xae, 0x08][..], Mnemonic::Fxrstor),
            (&[0x48, 0x0f, 0xae, 0x08][..], Mnemonic::Fxrstor64),
            (&[0x0f, 0xae, 0x28][..], Mnemonic::Xrstor),
            (&[0x48, 0x0f, 0xae, 0x28][..], Mnemonic::Xrstor64),
            (&[0x0f, 0xc7, 0x18][..], Mnemonic::Xrstors),
            (&[0x48, 0x0f, 0xc7, 0x18][..], Mnemonic::Xrstors64),
            (&[0xc5, 0xfc, 0x77][..], Mnemonic::Vzeroall),
            (&[0xc5, 0xf8, 0x77][..], Mnemonic::Vzeroupper),
        ] {
            assert_eq!(Decoder::new(64, code, DecoderOptions::NONE).decode().mnemonic(), mnemonic);
            assert_eq!(certificate(Vec::new(), code), ThunkKind::Full);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ChainSite {
    pub patch_offset: u32,
    pub target_pc: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct DirectSite {
    pub offset: u32,
    pub len: u8,
    pub slow_offset: u32,
}

const BITNESS: u32 = 64;

const DIRECT_SITE_MIN_LEN: usize = 3;

fn resolve_direct_sites(result: &CodeAssemblerResult) -> Result<Vec<DirectSite>> {
    use iced_x86::{Decoder, DecoderOptions, OpKind};
    let pending = crate::backend::isel::take_direct_sites();
    let code = &result.inner.code_buffer;
    let mut sites = Vec::with_capacity(pending.len());
    for site in pending {
        let offset = result
            .label_ip(&site.access)
            .map_err(|e| Error::Backend(e.to_string()))? as usize;
        let slow = site
            .slow
            .ok_or_else(|| Error::Backend("direct access site without a slow path".into()))?;
        let slow = result
            .label_ip(&slow)
            .map_err(|e| Error::Backend(e.to_string()))? as usize;
        let instruction = Decoder::new(BITNESS, &code[offset..], DecoderOptions::NONE).decode();
        let len = instruction.len();
        let accesses_memory = !instruction.is_invalid()
            && (0..instruction.op_count()).any(|i| instruction.op_kind(i) == OpKind::Memory);
        let rel = slow as isize - (offset as isize + 2);
        if !accesses_memory || len < DIRECT_SITE_MIN_LEN || i8::try_from(rel).is_err() {
            return Err(Error::Backend(format!(
                "direct access site at {offset:#x} (len {len}, slow path {slow:#x}) cannot be patched"
            )));
        }
        sites.push(DirectSite {
            offset: offset as u32,
            len: len as u8,
            slow_offset: slow as u32,
        });
    }
    Ok(sites)
}

/// Generated block layout:
///
/// ```text
/// entry:  cmp  word [ctx.should_halt], 0   ; should_halt | mem_fault
///         jne  bail
///         sub  qword [ctx.budget_remaining], N
///         js   bail_restore
/// body:   ...
/// exit:   jmp  rel32 (chain site, 0 = fall through)  ; patched to a block entry
///         mov  rax, target_pc
///         ret
/// bail_restore: add qword [ctx.budget_remaining], N
/// bail:   mov  rax, block_pc
///         ret
/// ```
///
/// Every entry (from the dispatcher, a chained jump or the indirect-branch
/// probe) charges the block's instruction count against the slice budget and
/// honours halt / fault requests, so chained execution stays bounded.  A
/// block that does not fit returns its own PC with the budget untouched.
pub fn emit_block(block: &Block) -> Result<EmittedBlock> {
    drop(crate::backend::isel::take_direct_sites());
    let fused = fusable_flag_producer(block);
    let fused_uses: [ValueRef; 2] = match fused {
        Some(p) => [p.args[0], p.args[1]],
        None => [ValueRef::NONE; 2],
    };
    let ranges = compute_live_ranges_with(
        block,
        if fused.is_some() { &fused_uses } else { &[] },
    );
    let alloc = linear_scan(block, &ranges, ALLOCATABLE_GPRS);
    if alloc.spill_bytes > crate::backend::operand::SPILL_LIMIT {
        // Translation retries with fewer instructions.
        return Err(Error::BlockTooLarge {
            pc: block.start_pc,
        });
    }

    let mut asm = CodeAssembler::new(BITNESS)?;

    emit_prologue(&mut asm, &alloc)?;

    let insns = block.cycles.max(1).min(i32::MAX as u32) as i32;
    let mut bail_label = asm.create_label();
    let mut bail_restore_label = asm.create_label();
    asm.cmp(word_ptr(CTX_REG + cpu_offsets::should_halt() as i32), 0)?;
    asm.jne(bail_label)?;
    asm.sub(
        qword_ptr(CTX_REG + cpu_offsets::budget_remaining() as i32),
        insns,
    )?;
    asm.js(bail_restore_label)?;

    let mut body_label = asm.create_label();
    asm.set_label(&mut body_label)?;
    asm.nop()?;

    for (vr, _) in block.iter_live() {
        emit_armlet(&mut asm, block, &alloc, vr.as_usize())?;
    }
    // Ops may end with a bound label (e.g. the join after a fastmem fast /
    // slow path); the terminal may start with one too (a chain site), and
    // iced rejects two labels on one instruction.
    asm.nop()?;

    let mut chain_specs: Vec<(CodeLabel, u64)> = Vec::new();
    let mut epilogue_label = asm.create_label();

    match block.terminal {
        Terminal::Invalid => {
            asm.mov(rax, block.end_pc as i64)?;
        }
        Terminal::LinkBlock { next_pc }
        | Terminal::DirectBranch {
            target_pc: next_pc,
            link: _,
        } => {
            let lbl = emit_patch_site(&mut asm, next_pc)?;
            chain_specs.push((lbl, next_pc));
        }
        Terminal::ConditionalBranch {
            cond_nzcv: _,
            cond_code,
            taken_pc,
            not_taken_pc,
        } => {
            let cond = Cond::from_bits(cond_code);
            let mut taken_label = asm.create_label();
            match fused {
                Some(producer) => {
                    emit_fused_flags(&mut asm, &alloc, producer)?;
                    emit_fused_jcc(&mut asm, producer.op, cond, taken_label)?;
                }
                None => {
                    // Truth-table lookup straight into a jump on CF.
                    let tt = crate::arch::COND_TRUTH[cond as usize] as i32;
                    asm.movzx(edx, byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32))?;
                    asm.mov(eax, tt)?;
                    asm.bt(eax, edx)?;
                    asm.jc(taken_label)?;
                }
            }
            emit_two_way_patches(
                &mut asm,
                &mut chain_specs,
                &mut taken_label,
                &mut epilogue_label,
                taken_pc,
                not_taken_pc,
            )?;
        }
        Terminal::CompareBranchZero {
            value,
            inverse,
            is_64,
            taken_pc,
            not_taken_pc,
        } => {
            load64(&mut asm, &alloc, value, rcx)?;
            if is_64 {
                asm.test(rcx, rcx)?;
            } else {
                asm.test(ecx, ecx)?;
            }
            let mut taken_label = asm.create_label();
            if inverse {
                asm.jnz(taken_label)?;
            } else {
                asm.jz(taken_label)?;
            }
            emit_two_way_patches(
                &mut asm,
                &mut chain_specs,
                &mut taken_label,
                &mut epilogue_label,
                taken_pc,
                not_taken_pc,
            )?;
        }
        Terminal::TestBranchBit {
            value,
            bit,
            inverse,
            taken_pc,
            not_taken_pc,
        } => {
            load64(&mut asm, &alloc, value, rcx)?;
            asm.bt(rcx, bit as u32 as i32)?;
            let mut taken_label = asm.create_label();
            if inverse {
                asm.jc(taken_label)?;
            } else {
                asm.jnc(taken_label)?;
            }
            emit_two_way_patches(
                &mut asm,
                &mut chain_specs,
                &mut taken_label,
                &mut epilogue_label,
                taken_pc,
                not_taken_pc,
            )?;
        }
        Terminal::IndirectBranch {
            target,
            link: _,
            is_ret: _,
        } => {
            load64(&mut asm, &alloc, target, rax)?;
            if block.link_indirect {
                emit_dispatch_probe(&mut asm, epilogue_label)?;
            }
        }
        Terminal::Exception { kind, imm } => {
            let kind_byte: u64 = match kind {
                ExceptionKind::Svc => 0x01,
                ExceptionKind::Brk => 0x02,
                ExceptionKind::Hvc => 0x03,
                ExceptionKind::UnknownInst => 0xFF,
            };
            let exit_value: u64 = 0xE000_0000_0000_0000 | ((imm as u64 & 0xFFFF) << 8) | kind_byte;
            // SVC / HVC return to the next instruction; a breakpoint's
            // preferred return address is the BRK itself (as Unicorn and
            // dynarmic report it).
            let pc_after = if matches!(kind, ExceptionKind::Brk) {
                block.end_pc.wrapping_sub(4)
            } else {
                block.end_pc
            };
            asm.mov(rcx, pc_after as i64)?;
            asm.mov(qword_ptr(CTX_REG + cpu_offsets::pc() as i32), rcx)?;
            asm.mov(rax, exit_value as i64)?;
        }
    }

    asm.nop()?;
    asm.set_label(&mut epilogue_label)?;
    emit_epilogue(&mut asm, &alloc)?;

    asm.set_label(&mut bail_restore_label)?;
    asm.add(
        qword_ptr(CTX_REG + cpu_offsets::budget_remaining() as i32),
        insns,
    )?;
    asm.set_label(&mut bail_label)?;
    asm.mov(rax, block.start_pc as i64)?;
    emit_epilogue(&mut asm, &alloc)?;

    let result = asm
        .assemble_options(0, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS)
        .map_err(|e| Error::Backend(e.to_string()))?;

    let mut chains = Vec::with_capacity(chain_specs.len());
    for (lbl, target_pc) in chain_specs {
        let off = result
            .label_ip(&lbl)
            .map_err(|e| Error::Backend(e.to_string()))?;
        chains.push(ChainSite {
            patch_offset: off as u32,
            target_pc,
        });
    }
    let body_offset = result
        .label_ip(&body_label)
        .map_err(|e| Error::Backend(e.to_string()))? as u32;
    let direct_sites = resolve_direct_sites(&result)?;

    let code = result.inner.code_buffer;
    let thunk_kind = emitted_thunk_kind(block, &alloc, &code);
    Ok(EmittedBlock {
        code,
        chains,
        direct_sites,
        body_offset,
        thunk_kind,
    })
}

/// For a block ending in `b.cond`, returns the armlet whose flags the branch
/// consumes when that is a plain 32/64-bit ADDS/SUBS (CMP/CMN/...) the
/// terminal can re-execute on host flags, i.e. it is the last armlet in the
/// block that may write NZCV and the condition maps to one x86 `jcc`.  The
/// NZCV byte is still materialised by the producer for later blocks.
fn fusable_flag_producer(block: &Block) -> Option<Armlet> {
    let Terminal::ConditionalBranch { cond_code, .. } = block.terminal else {
        return None;
    };
    let mut producer: Option<Armlet> = None;
    for (_, a) in block.iter_live() {
        match a.op {
            Op::AddsFlags32
            | Op::AddsFlags64
            | Op::SubsFlags32
            | Op::SubsFlags64
            | Op::LogicFlags32
            | Op::LogicFlags64 => {
                producer = Some(*a);
            }
            op if !op_preserves_nzcv(op) => producer = None,
            _ => {}
        }
    }
    let p = producer?;
    let is_logic = matches!(p.op, Op::LogicFlags32 | Op::LogicFlags64);
    if p.args[0].is_none() || (!is_logic && p.args[1].is_none()) {
        return None;
    }
    fused_jcc_kind(p.op, Cond::from_bits(cond_code)).map(|_| p)
}

/// Ops known not to write the guest NZCV byte.  Everything else (SetNzcv,
/// FCMP, MSR, ...) ends a fusion candidate.
fn op_preserves_nzcv(op: Op) -> bool {
    use Op::*;
    op.is_pure()
        || op.is_terminator()
        || matches!(
            op,
            Void | Identity
                | GetX
                | GetW
                | GetSp
                | GetNzcv
                | GetPc
                | GetV
                | SetX
                | SetW
                | SetSp
                | SetV
                | Load8
                | Load16
                | Load32
                | Load64
                | Load128
                | LoadS8
                | LoadS16
                | LoadS32
                | Store8
                | Store16
                | Store32
                | Store64
                | Store128
                | LoadEx8
                | LoadEx16
                | LoadEx32
                | LoadEx64
                | LoadEx128
                | StoreEx8
                | StoreEx16
                | StoreEx32
                | StoreEx64
                | StoreEx128
                | AtomicRmw8
                | AtomicRmw16
                | AtomicRmw32
                | AtomicRmw64
                | AtomicCas8
                | AtomicCas16
                | AtomicCas32
                | AtomicCas64
                | AtomicCas128
        )
}

#[derive(Clone, Copy)]
enum Jcc {
    E,
    Ne,
    Ae,
    B,
    S,
    Ns,
    O,
    No,
    A,
    Be,
    Ge,
    L,
    G,
    Le,
}

/// x86 condition equivalent to ARM `cond` after re-executing `op` (x86 ADD /
/// SUB flags).  ARM C is x86 CF for ADDS and !CF (no borrow) for SUBS.
/// Logical producers re-execute as x86 `test`, which (like ARM ANDS) clears
/// CF and OF, so CS/HI are never taken and CC/LS always are; `jb` / `jae`
/// encode exactly that on CF = 0.
fn fused_jcc_kind(op: Op, cond: Cond) -> Option<Jcc> {
    if matches!(op, Op::LogicFlags32 | Op::LogicFlags64) {
        return Some(match cond as u8 {
            0 => Jcc::E,
            1 => Jcc::Ne,
            2 | 8 => Jcc::B,
            3 | 9 => Jcc::Ae,
            4 => Jcc::S,
            5 => Jcc::Ns,
            6 => Jcc::O,
            7 => Jcc::No,
            10 => Jcc::Ge,
            11 => Jcc::L,
            12 => Jcc::G,
            13 => Jcc::Le,
            _ => return None,
        });
    }
    let is_sub = matches!(op, Op::SubsFlags32 | Op::SubsFlags64);
    Some(match cond as u8 {
        0 => Jcc::E,
        1 => Jcc::Ne,
        2 => if is_sub { Jcc::Ae } else { Jcc::B },
        3 => if is_sub { Jcc::B } else { Jcc::Ae },
        4 => Jcc::S,
        5 => Jcc::Ns,
        6 => Jcc::O,
        7 => Jcc::No,
        8 if is_sub => Jcc::A,
        9 if is_sub => Jcc::Be,
        10 => Jcc::Ge,
        11 => Jcc::L,
        12 => Jcc::G,
        13 => Jcc::Le,
        _ => return None,
    })
}

fn emit_fused_flags(asm: &mut CodeAssembler, alloc: &crate::backend::regalloc::Allocation, p: Armlet) -> Result<()> {
    use crate::backend::operand::load32;
    let is_sub = matches!(p.op, Op::SubsFlags32 | Op::SubsFlags64);
    use crate::backend::isel::imm_operand;
    match p.op {
        Op::LogicFlags64 => {
            load64(asm, alloc, p.args[0], rax)?;
            asm.test(rax, rax)?;
            return Ok(());
        }
        Op::LogicFlags32 => {
            load32(asm, alloc, p.args[0], eax)?;
            asm.test(eax, eax)?;
            return Ok(());
        }
        _ => {}
    }
    if matches!(p.op, Op::AddsFlags64 | Op::SubsFlags64) {
        load64(asm, alloc, p.args[0], rax)?;
        match imm_operand(alloc, p.args[1], 64) {
            Some(imm) if is_sub => asm.cmp(rax, imm)?,
            Some(imm) => asm.add(rax, imm)?,
            None => {
                load64(asm, alloc, p.args[1], rcx)?;
                if is_sub {
                    asm.cmp(rax, rcx)?;
                } else {
                    asm.add(rax, rcx)?;
                }
            }
        }
    } else {
        load32(asm, alloc, p.args[0], eax)?;
        match imm_operand(alloc, p.args[1], 32) {
            Some(imm) if is_sub => asm.cmp(eax, imm)?,
            Some(imm) => asm.add(eax, imm)?,
            None => {
                load32(asm, alloc, p.args[1], ecx)?;
                if is_sub {
                    asm.cmp(eax, ecx)?;
                } else {
                    asm.add(eax, ecx)?;
                }
            }
        }
    }
    Ok(())
}

fn emit_fused_jcc(asm: &mut CodeAssembler, op: Op, cond: Cond, taken: CodeLabel) -> Result<()> {
    let kind = fused_jcc_kind(op, cond).ok_or_else(|| Error::Backend("unfusable condition".into()))?;
    match kind {
        Jcc::E => asm.je(taken)?,
        Jcc::Ne => asm.jne(taken)?,
        Jcc::Ae => asm.jae(taken)?,
        Jcc::B => asm.jb(taken)?,
        Jcc::S => asm.js(taken)?,
        Jcc::Ns => asm.jns(taken)?,
        Jcc::O => asm.jo(taken)?,
        Jcc::No => asm.jno(taken)?,
        Jcc::A => asm.ja(taken)?,
        Jcc::Be => asm.jbe(taken)?,
        Jcc::Ge => asm.jge(taken)?,
        Jcc::L => asm.jl(taken)?,
        Jcc::G => asm.jg(taken)?,
        Jcc::Le => asm.jle(taken)?,
    }
    Ok(())
}

/// A chain site: `jmp rel32` whose displacement is 0 (fall through to the
/// return-to-dispatcher path) until the code cache links it.
const CHAIN_SITE_UNLINKED: [u8; 5] = [0xE9, 0, 0, 0, 0];

fn emit_patch_site(asm: &mut CodeAssembler, fallback_pc: u64) -> Result<CodeLabel> {
    let mut label = asm.create_label();
    emit_patch_site_at_label(asm, &mut label, fallback_pc)?;
    Ok(label)
}

fn emit_patch_site_at_label(
    asm: &mut CodeAssembler,
    label: &mut CodeLabel,
    fallback_pc: u64,
) -> Result<()> {
    asm.set_label(label)?;
    asm.db(&CHAIN_SITE_UNLINKED)?;
    asm.mov(rax, fallback_pc as i64)?;
    Ok(())
}

/// Probes the direct-mapped dispatch table with the guest target in RAX and
/// jumps straight to the block on a hit.  Misses return RAX to the
/// dispatcher.  The target block's entry performs the halt and budget
/// checks.
fn emit_dispatch_probe(asm: &mut CodeAssembler, miss: CodeLabel) -> Result<()> {
    use crate::jit::code_cache::DISPATCH_MASK;
    const _: () = assert!(core::mem::size_of::<crate::jit::code_cache::DispatchSlot>() == 16);
    use crate::jit::code_cache::DISPATCH_BITS;
    asm.mov(rcx, rax)?;
    asm.shr(rcx, DISPATCH_BITS)?;
    asm.xor(rcx, rax)?;
    asm.shr(rcx, 2)?;
    asm.and(ecx, DISPATCH_MASK as i32)?;
    asm.shl(ecx, 4)?;
    asm.add(rcx, qword_ptr(CTX_REG + cpu_offsets::dispatch_table() as i32))?;
    asm.cmp(qword_ptr(rcx), rax)?;
    asm.jne(miss)?;
    asm.jmp(qword_ptr(rcx + 8))?;
    Ok(())
}

fn emit_two_way_patches(
    asm: &mut CodeAssembler,
    chain_specs: &mut Vec<(CodeLabel, u64)>,
    taken_label: &mut CodeLabel,
    epilogue_label: &mut CodeLabel,
    taken_pc: u64,
    not_taken_pc: u64,
) -> Result<()> {
    let lbl_nt = emit_patch_site(asm, not_taken_pc)?;
    chain_specs.push((lbl_nt, not_taken_pc));
    asm.jmp(*epilogue_label)?;

    emit_patch_site_at_label(asm, taken_label, taken_pc)?;
    chain_specs.push((*taken_label, taken_pc));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::{TranslateOptions, translate_block_into};
    use crate::optimizer::{Scratch, optimize_with_scratch};

    fn block_of(words: &[u32]) -> Block {
        let mut block = Block::new(0);
        translate_block_into(
            &mut block,
            0x1000,
            &mut |a| words.get(((a - 0x1000) / 4) as usize).copied(),
            TranslateOptions::default(),
        )
        .unwrap();
        optimize_with_scratch(&mut block, &mut Scratch::new());
        block
    }

    #[test]
    fn compare_branch_is_fused_only_when_nothing_else_writes_nzcv() {
        // cmp x0, x1 ; add x2, x2, #1 ; b.lt +8
        let b = block_of(&[0xEB01_001F, 0x9100_0442, 0x5400_004B]);
        assert!(fusable_flag_producer(&b).is_some());
        emit_block(&b).unwrap();
        // cmn w0, w1 ; b.hi: no single x86 condition for ADDS + HI.
        let b = block_of(&[0x2B01_001F, 0x5400_0048]);
        assert!(fusable_flag_producer(&b).is_none());
        // cmp x0, x1 ; fcmp d0, d1 ; b.eq: FCMP is the producer.
        let b = block_of(&[0xEB01_001F, 0x1E61_2000, 0x5400_0040]);
        assert!(fusable_flag_producer(&b).is_none());
        // cmp ; msr nzcv, x3 ; b.eq
        let b = block_of(&[0xEB01_001F, 0xD51B_4203, 0x5400_0040]);
        assert!(fusable_flag_producer(&b).is_none());
    }
}
