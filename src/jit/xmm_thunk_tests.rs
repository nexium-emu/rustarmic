//! The wrapper seeds and compares every bit of all ten Win64 nonvolatile
//! vector registers, while preserving its caller's original registers itself.
use super::*;
use crate::backend::emit::emit_block;
use crate::backend::regalloc::{ALLOCATABLE_GPRS, Loc, compute_live_ranges, linear_scan};
use crate::frontend::{TranslateOptions, translate_block_into};
use crate::ir::{Block, IrEmitter, Terminal};
use crate::jit::context::CpuContext;
use crate::optimizer::{Scratch, optimize_with_scratch};
use iced_x86::code_asm::*;

const PC: u64 = 0x1000;
const GUARD: u64 = 0xa1b2_c3d4_5678_9012;

#[test]
fn lean_thunk_omits_only_the_twenty_nonvolatile_vector_moves() {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};
    let full = emit_thunk_bytes_for(ThunkKind::Full).unwrap();
    let lean = emit_thunk_bytes_for(ThunkKind::XmmFree).unwrap();
    let mut decoder = Decoder::new(64, &full, DecoderOptions::NONE);
    let mut stripped = Vec::new();
    let mut vector_moves = 0;
    while decoder.can_decode() {
        let start = decoder.position();
        let instruction = decoder.decode();
        assert!(!instruction.is_invalid());
        if instruction.mnemonic() == Mnemonic::Movdqu {
            vector_moves += 1;
        } else {
            stripped.extend_from_slice(&full[start..decoder.position()]);
        }
    }
    assert_eq!(vector_moves, 20);
    assert_eq!(stripped, lean, "frame/GPR/argument/call bytes changed");
}

fn vector_register(index: i32) -> AsmRegisterXmm {
    [xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15][index as usize]
}

fn sentinel_wrapper() -> Allocation {
    // RCX=block, RDX=context, R8=thunk, R9=seeds/output. Two pushes and a
    // 200-byte frame retain Win64 call alignment and 32 bytes of shadow space.
    let mut asm = CodeAssembler::new(64).unwrap();
    asm.push(rbx).unwrap();
    asm.push(r12).unwrap();
    asm.sub(rsp, 200).unwrap();
    asm.mov(rbx, r9).unwrap();
    asm.mov(r12, r8).unwrap();
    for index in 0..10i32 {
        asm.movdqu(xmmword_ptr(rsp + 32 + index * 16), vector_register(index)).unwrap();
        asm.movdqu(vector_register(index), xmmword_ptr(rbx + index * 16)).unwrap();
    }
    asm.call(r12).unwrap();
    for index in 0..10i32 {
        asm.movdqu(xmmword_ptr(rbx + 160 + index * 16), vector_register(index)).unwrap();
        asm.movdqu(vector_register(index), xmmword_ptr(rsp + 32 + index * 16)).unwrap();
    }
    asm.add(rsp, 200).unwrap();
    asm.pop(r12).unwrap();
    asm.pop(rbx).unwrap();
    asm.ret().unwrap();
    let bytes = asm.assemble(0).unwrap();
    let mut allocation = region::alloc(bytes.len(), Protection::READ_WRITE).unwrap();
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), allocation.as_mut_ptr(), bytes.len());
        region::protect(allocation.as_ptr::<u8>(), allocation.len(), Protection::READ_EXECUTE).unwrap();
    }
    allocation
}

fn guarded_call(cache: &CodeCache, block: *const u8, kind: ThunkKind, context: &mut CpuContext) -> u64 {
    type Wrapper = unsafe extern "C" fn(u64, *mut CpuContext, u64, *mut [[u64; 2]; 20]) -> u64;
    let allocation = sentinel_wrapper();
    let wrapper: Wrapper = unsafe { core::mem::transmute(allocation.as_ptr::<u8>()) };
    let mut patterns = [[0u64; 2]; 20];
    for (index, pattern) in patterns[..10].iter_mut().enumerate() {
        *pattern = [GUARD.rotate_left(index as u32 * 5), (!GUARD).wrapping_add(index as u64 * 0x12345)];
    }
    patterns[10..].fill([0xdeaf_dead_bad0_beef; 2]);
    let expected = patterns[..10].to_vec();
    let result = unsafe { wrapper(block as u64, context, cache.thunk_for(kind) as u64, &mut patterns) };
    assert_eq!(&patterns[..10], expected.as_slice(), "wrapper seeds were overwritten");
    assert_eq!(&patterns[10..], expected.as_slice(), "Win64 XMM6..15 were not preserved");
    result
}

fn seeded_context() -> CpuContext {
    let mut context = CpuContext::default();
    context.x = [GUARD; 31];
    context.v = [[GUARD, !GUARD]; 32];
    context.pc = PC;
    context.sp = GUARD;
    context.nzcv = 0xb;
    context.fpcr = 0x0300_0000;
    context.fpsr = 0x0800_009f;
    context
}

fn status_guards(context: &CpuContext) {
    assert_eq!((context.sp, context.nzcv, context.fpcr, context.fpsr),
        (GUARD, 0xb, 0x0300_0000, 0x0800_009f));
}

fn emit(words: &[u32]) -> crate::backend::EmittedBlock {
    let mut block = Block::new(PC);
    translate_block_into(&mut block, PC, &mut |pc| words.get(((pc - PC) / 4) as usize).copied(),
        TranslateOptions { max_insts: words.len() as u32, multiblock: false }).unwrap();
    optimize_with_scratch(&mut block, &mut Scratch::new());
    emit_block(&block).unwrap()
}

fn install(cache: &mut CodeCache, emitted: &crate::backend::EmittedBlock, normal: bool, count: u32) -> *const u8 {
    cache.install_emitted(PC, PC + u64::from(count) * 4, count, &emitted.code,
        &emitted.chains, emitted.body_offset, normal, emitted.thunk_kind).unwrap()
}

#[test]
fn raw_public_install_uses_full_thunk_and_restores_every_nonvolatile_vector() {
    let mut assembler = CodeAssembler::new(64).unwrap();
    for index in 0..10 {
        assembler.pxor(vector_register(index), vector_register(index)).unwrap();
    }
    assembler.mov(eax, 47u32).unwrap();
    assembler.ret().unwrap();
    let mut cache = CodeCache::new(64 * 1024).unwrap();
    let pointer = cache.install(PC, PC + 4, 1, &assembler.assemble(0).unwrap(), &[], 0, true).unwrap();
    let descriptor = cache.lookup_dispatch_meta(PC).unwrap();
    assert_eq!(descriptor.thunk_kind, ThunkKind::Full);
    assert_eq!(cache.thunk_for(descriptor.thunk_kind), cache.thunk());
    let mut context = seeded_context();
    assert_eq!(guarded_call(&cache, pointer, descriptor.thunk_kind, &mut context), 47);
    assert_eq!(context.x, [GUARD; 31]);
    assert_eq!(context.v, [[GUARD, !GUARD]; 32]);
    status_guards(&context);
}

#[test]
fn emitted_integer_and_vector_blocks_choose_safe_thunks_with_full_host_guards() {
    let mut cache = CodeCache::new(64 * 1024).unwrap();
    let simple = emit(&[0x9100_0400]); // ADD X0,X0,#1.
    assert_eq!(simple.thunk_kind, ThunkKind::XmmFree);
    let pointer = install(&mut cache, &simple, true, 1);
    let mut context = seeded_context();
    assert_eq!(guarded_call(&cache, pointer, simple.thunk_kind, &mut context), PC + 4);
    let mut expected_x = [GUARD; 31];
    expected_x[0] = GUARD + 1;
    assert_eq!(context.x, expected_x);
    assert_eq!(context.v, [[GUARD, !GUARD]; 32]);
    status_guards(&context);

    let vector = emit(&[0x4ea2_8420]); // ADD V0.4S,V1.4S,V2.4S.
    assert_eq!(vector.thunk_kind, ThunkKind::Full);
    let pointer = install(&mut cache, &vector, true, 1);
    let mut context = seeded_context();
    assert_eq!(guarded_call(&cache, pointer, vector.thunk_kind, &mut context), PC + 4);
    let lane = (GUARD as u32).wrapping_mul(2) as u64;
    let high_lane = ((GUARD >> 32) as u32).wrapping_mul(2) as u64;
    let lane_inv = ((!GUARD) as u32).wrapping_mul(2) as u64;
    let high_inv = (((!GUARD) >> 32) as u32).wrapping_mul(2) as u64;
    let mut expected_v = [[GUARD, !GUARD]; 32];
    expected_v[0] = [lane | (high_lane << 32), lane_inv | (high_inv << 32)];
    assert_eq!(context.x, [GUARD; 31]);
    assert_eq!(context.v, expected_v);
    status_guards(&context);
}

#[test]
fn historical_pressure_uses_full_thunk_even_when_spills_expire() {
    let mut block = Block::new(PC);
    let mut emitter = IrEmitter::new(&mut block, PC);
    let values: Vec<_> = (0..16).map(|register| emitter.get_x(register)).collect();
    for (register, value) in values.into_iter().enumerate() {
        emitter.set_x(register as u8 + 15, value);
    }
    let final_value = emitter.const_u64(GUARD);
    emitter.set_x(0, final_value);
    block.end_pc = PC + 4;
    block.terminal = Terminal::LinkBlock { next_pc: PC + 4 };
    let allocation = linear_scan(&block, &compute_live_ranges(&block), ALLOCATABLE_GPRS);
    assert!(allocation.locs.iter().any(|location| matches!(location, Loc::Xmm(6..=15))));
    assert_eq!(allocation.used_xmms, 0, "historical integer spills should have expired");
    let emitted = emit_block(&block).unwrap();
    assert_eq!(emitted.thunk_kind, ThunkKind::Full);
    let mut cache = CodeCache::new(64 * 1024).unwrap();
    let pointer = install(&mut cache, &emitted, true, 1);
    let mut context = seeded_context();
    for register in 0..31 { context.x[register] ^= register as u64 * 123; }
    let mut expected_x = context.x;
    expected_x[15..31].copy_from_slice(&context.x[..16]);
    assert_eq!(guarded_call(&cache, pointer, emitted.thunk_kind, &mut context), PC + 4);
    assert_eq!(context.x, expected_x);
    assert_eq!(context.v, [[GUARD, !GUARD]; 32]);
    status_guards(&context);
}

unsafe extern "C" fn volatile_read(context: *mut CpuContext, address: u64, size: u8) {
    unsafe {
        core::arch::asm!("pxor xmm0, xmm0", "pxor xmm1, xmm1", "pxor xmm2, xmm2",
            "pxor xmm3, xmm3", "pxor xmm4, xmm4", "pxor xmm5, xmm5",
            out("xmm0") _, out("xmm1") _, out("xmm2") _, out("xmm3") _, out("xmm4") _, out("xmm5") _,
            options(nostack));
        (*context).io_value = [address ^ u64::from(size), 0];
        (*context).core_id += 1;
    }
}

#[test]
fn lean_callback_block_preserves_host_vectors_while_helpers_clobber_all_volatile_vectors() {
    let emitted = emit(&[0xf940_0020]); // LDR X0,[X1].
    assert_eq!(emitted.thunk_kind, ThunkKind::XmmFree);
    let mut cache = CodeCache::new(64 * 1024).unwrap();
    let pointer = install(&mut cache, &emitted, true, 1);
    let mut context = seeded_context();
    context.mem_read = volatile_read;
    assert_eq!(guarded_call(&cache, pointer, emitted.thunk_kind, &mut context), PC + 4);
    let mut expected_x = [GUARD; 31];
    expected_x[0] ^= 8;
    assert_eq!(context.x, expected_x);
    assert_eq!(context.v, [[GUARD, !GUARD]; 32]);
    assert_eq!((context.core_id, context.mem_fault, context.should_halt), (1, 0, 0));
    status_guards(&context);
}

#[test]
fn cached_descriptor_replacement_transient_collision_reset_and_append_failure_keep_kind_paired() {
    let mut cache = CodeCache::new(64 * 1024).unwrap();
    let lean = emit(&[0x9100_0400]);
    let first = install(&mut cache, &lean, true, 7);
    assert_eq!(cache.lookup_dispatch_meta(PC), Some(DispatchMeta { host_ptr: first, guest_insns: 7, thunk_kind: ThunkKind::XmmFree }));
    let full = emit(&[0x4ea2_8420]);
    let second = install(&mut cache, &full, true, 3);
    assert_eq!(cache.lookup_dispatch_meta(PC), Some(DispatchMeta { host_ptr: second, guest_insns: 3, thunk_kind: ThunkKind::Full }));
    let transient = install(&mut cache, &lean, false, 1);
    assert_ne!(second, transient);
    let mut context = seeded_context();
    assert_eq!(guarded_call(&cache, transient, lean.thunk_kind, &mut context), PC + 4);
    assert_eq!(cache.lookup_dispatch_meta(PC).unwrap().host_ptr, second);
    assert_eq!(cache.lookup_dispatch_meta(PC).unwrap().thunk_kind, ThunkKind::Full);
    let collision_pc = PC | (1u64 << 56);
    let collision = cache.install_emitted(collision_pc, collision_pc + 4, 11, &[0xc3], &[], 0, true, ThunkKind::XmmFree).unwrap();
    for _ in 0..10 {
        assert_eq!(cache.lookup_dispatch_meta(collision_pc).unwrap(), DispatchMeta { host_ptr: collision, guest_insns: 11, thunk_kind: ThunkKind::XmmFree });
        assert_eq!(cache.lookup_dispatch_meta(PC).unwrap(), DispatchMeta { host_ptr: second, guest_insns: 3, thunk_kind: ThunkKind::Full });
    }
    let remaining = cache.capacity - ((cache.cursor + 15) & !15);
    cache.append_raw(&vec![0x90; remaining]).unwrap();
    assert!(matches!(cache.install_emitted(PC, PC + 4, 99, &[0xc3], &[], 0, true, ThunkKind::XmmFree), Err(Error::CodeCacheFull)));
    assert_eq!(cache.lookup_dispatch_meta(PC).unwrap(), DispatchMeta { host_ptr: second, guest_insns: 3, thunk_kind: ThunkKind::Full });
    cache.invalidate_range(PC + 4, 4);
    assert_eq!(cache.lookup_dispatch_meta(PC), None);
    assert_eq!(cache.lookup_dispatch_meta(collision_pc).unwrap().host_ptr, collision);
    cache.invalidate_range(u64::MAX - 1, 8);
    assert_eq!(cache.lookup_dispatch_meta(collision_pc), None);
    cache.reset().unwrap();
    let replacement = install(&mut cache, &lean, true, 1);
    assert_eq!(replacement, first);
    let mut context = seeded_context();
    assert_eq!(guarded_call(&cache, replacement, lean.thunk_kind, &mut context), PC + 4);
    assert_eq!(cache.lookup_dispatch_meta(PC).unwrap().thunk_kind, ThunkKind::XmmFree);
}
