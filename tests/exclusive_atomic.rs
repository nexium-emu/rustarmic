//! Store-exclusive is a value compare-and-swap through `CpuContext::mem_atomic`
//! (`AtomicOp::Cas`); upstream wrote these tests against a separate
//! `mem_write_exclusive` callback with the same semantics.

use rustarmic::jit::atomic::AtomicOp;
use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};

const CODE: u64 = 0x1000;
const DATA: u64 = 0x20_0000;

struct CodeMem(Vec<u32>);

impl Memory for CodeMem {
    fn fetch_inst(&mut self, address: u64) -> Option<u32> {
        let offset = address.checked_sub(CODE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get((offset / 4) as usize).copied()
    }
}

fn mask(size: u8) -> u64 {
    match size {
        1 => u8::MAX as u64,
        2 => u16::MAX as u64,
        4 => u32::MAX as u64,
        8 => u64::MAX,
        _ => unreachable!(),
    }
}

unsafe fn backing(context: *mut CpuContext) -> &'static AtomicU64 {
    unsafe { &*((*context).mem_base as *const AtomicU64) }
}

unsafe extern "C" fn read(context: *mut CpuContext, address: u64, size: u8) {
    assert_eq!(address, DATA);
    let value = unsafe { backing(context) }.load(Ordering::SeqCst);
    // Poison unconsumed upper bits to verify the backend snapshots the actual
    // zero-extended LDXRB/H/W result, rather than the raw callback scratch word.
    unsafe { (*context).io_value[0] = (value & mask(size)) | !mask(size) };
}

unsafe extern "C" fn write(context: *mut CpuContext, address: u64, size: u8) {
    assert_eq!(address, DATA);
    let cell = unsafe { backing(context) };
    let bits = unsafe { (*context).io_value[0] } & mask(size);
    loop {
        let old = cell.load(Ordering::SeqCst);
        let new = (old & !mask(size)) | bits;
        if cell
            .compare_exchange(old, new, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            break;
        }
    }
}

/// `mem_atomic` compare-and-swap: replaces the `size`-byte value at `address`
/// with `io_value[0]` only if it equals `atomic_expected[0]`, returning the
/// observed value in `io_value[0]`.
unsafe extern "C" fn exclusive_write(context: *mut CpuContext, address: u64, size: u8, op: u8) {
    assert_eq!(address, DATA);
    assert_eq!(op, AtomicOp::Cas as u8);
    let cell = unsafe { backing(context) };
    let expected = unsafe { (*context).atomic_expected[0] } & mask(size);
    let desired = unsafe { (*context).io_value[0] } & mask(size);
    let mut current = cell.load(Ordering::SeqCst);
    while current & mask(size) == expected {
        let replacement = (current & !mask(size)) | desired;
        match cell.compare_exchange(current, replacement, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
    unsafe { (*context).io_value = [current & mask(size), 0] };
}

fn context(cell: &AtomicU64) -> CpuContext {
    let mut context = CpuContext::default();
    context.pc = CODE;
    context.x[1] = DATA;
    context.mem_base = (cell as *const AtomicU64).cast_mut().cast();
    context.mem_base_va = DATA;
    context.mem_size = 8;
    context.mem_read = read;
    context.mem_write = write;
    context.mem_atomic = exclusive_write;
    context
}

fn words(bytes: u32, acquire_release: bool) -> Vec<u32> {
    let width = bytes.trailing_zeros() << 30;
    let ordered = if acquire_release { 1 << 15 } else { 0 };
    vec![
        0x085f_7c20 | width | ordered,
        0x0802_7c23 | width | ordered,
        0xd420_0000,
    ] // LDXR W/X0,[X1]; STXR W2,W/X3,[X1]; BRK.
}

fn step(jit: &mut Jit, context: &mut CpuContext, memory: &mut CodeMem) {
    let result = jit.run_bounded(context, memory, 1).unwrap();
    assert_eq!(result.stop, StopReason::BudgetExhausted);
    assert_eq!(result.retired, 1);
}

#[test]
fn two_cores_cannot_both_store_a_stale_loaded_value() {
    for bytes in [1, 2, 4, 8] {
        for ordered in [false, true] {
            for fastmem in [false, true] {
                let cell = AtomicU64::new(0);
                let mut a = context(&cell);
                let mut b = context(&cell);
                a.x[3] = 0x11;
                b.x[3] = 0x22;
                let mut a_memory = CodeMem(words(bytes, ordered));
                let mut b_memory = CodeMem(words(bytes, ordered));
                let mut a_jit = Jit::new(JitConfig {
                    use_fastmem: fastmem,
                    ..JitConfig::default()
                })
                .unwrap();
                let mut b_jit = Jit::new(JitConfig {
                    use_fastmem: fastmem,
                    ..JitConfig::default()
                })
                .unwrap();
                step(&mut a_jit, &mut a, &mut a_memory);
                step(&mut b_jit, &mut b, &mut b_memory);
                assert_eq!(a.exclusive_value, 0);
                assert_eq!(b.exclusive_value, 0);
                step(&mut a_jit, &mut a, &mut a_memory);
                step(&mut b_jit, &mut b, &mut b_memory);
                assert_eq!(a.x[2], 0);
                assert_eq!(b.x[2], 1);
                assert_eq!(a.exclusive_size, 0);
                assert_eq!(b.exclusive_size, 0);
                assert_eq!(cell.load(Ordering::SeqCst), 0x11);
            }
        }
    }
}

#[test]
fn ordinary_intervening_write_invalidates_value_cas_and_second_attempt() {
    for bytes in [1, 2, 4, 8] {
        let cell = AtomicU64::new(0xa5a5_a5a5_a5a5_a5a5);
        let mut context = context(&cell);
        context.x[3] = 0x12;
        let mut memory = CodeMem(words(bytes, false));
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        step(&mut jit, &mut context, &mut memory);
        assert_eq!(
            context.exclusive_value,
            cell.load(Ordering::SeqCst) & mask(bytes as u8)
        );
        cell.store(0x3131_3131_3131_3131, Ordering::SeqCst);
        step(&mut jit, &mut context, &mut memory);
        assert_eq!(context.x[2], 1);
        assert_eq!(cell.load(Ordering::SeqCst), 0x3131_3131_3131_3131);
        assert_eq!(context.exclusive_size, 0);
        context.pc = CODE + 4;
        step(&mut jit, &mut context, &mut memory);
        assert_eq!(
            context.x[2], 1,
            "failed STXR consumes the local reservation"
        );
    }
}

#[test]
fn concurrent_atomic_increment_has_no_lost_updates() {
    const INCREMENTS: u64 = 2000;
    let cell = Arc::new(AtomicU64::new(0));
    let barrier = Arc::new(Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let cell = cell.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let mut context = context(&cell);
                let mut memory = CodeMem(words(8, true));
                let mut jit = Jit::new(JitConfig::default()).unwrap();
                barrier.wait();
                for _ in 0..INCREMENTS {
                    loop {
                        context.pc = CODE;
                        step(&mut jit, &mut context, &mut memory);
                        context.x[3] = context.x[0] + 1;
                        step(&mut jit, &mut context, &mut memory);
                        if context.x[2] == 0 {
                            break;
                        }
                    }
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    assert_eq!(cell.load(Ordering::SeqCst), INCREMENTS * 2);
}

#[test]
fn concurrent_guest_retry_loops_complete_without_lost_updates() {
    const INCREMENTS: u64 = 10_000;
    let cell = Arc::new(AtomicU64::new(0));
    let barrier = Arc::new(Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let cell = cell.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let mut context = context(&cell);
                context.x[4] = INCREMENTS;
                let mut memory = CodeMem(vec![
                    0xc85f_fc20, // LDAXR X0,[X1].
                    0x9100_0403, // ADD X3,X0,#1.
                    0xc802_fc23, // STLXR W2,X3,[X1].
                    0x35ff_ffa2, // CBNZ W2, -12 (retry LDAXR).
                    0xf100_0484, // SUBS X4,X4,#1.
                    0x54ff_ff61, // B.NE -20 (next increment).
                    0xd420_0000, // BRK #0.
                ]);
                let mut jit = Jit::new(JitConfig::default()).unwrap();
                barrier.wait();
                let result = jit
                    .run_bounded(&mut context, &mut memory, 1_000_000)
                    .unwrap();
                assert_eq!(result.stop, StopReason::Brk(0));
                assert_eq!(context.x[4], 0);
                assert_eq!(context.exclusive_size, 0);
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
    assert_eq!(cell.load(Ordering::SeqCst), INCREMENTS * 2);
}

unsafe extern "C" fn denied_write(context: *mut CpuContext, address: u64, size: u8, _op: u8) {
    unsafe {
        (*context).io_value = [0, 0];
        (*context).mem_fault = 1;
        (*context).mem_fault_access = 1;
        (*context).mem_fault_cause = 1;
        (*context).mem_fault_addr = address;
        (*context).mem_fault_size = size;
        (*context).mem_fault_pc = (*context).pc;
    }
}

unsafe extern "C" fn denied_read(context: *mut CpuContext, address: u64, size: u8) {
    unsafe {
        (*context).io_value = [0, 0];
        (*context).mem_fault = 1;
        (*context).mem_fault_access = 0;
        (*context).mem_fault_cause = 0;
        (*context).mem_fault_addr = address;
        (*context).mem_fault_size = size;
        (*context).mem_fault_pc = (*context).pc;
    }
}

#[test]
fn exclusive_callback_write_fault_is_precise_and_consumes_reservation() {
    let cell = AtomicU64::new(7);
    let mut context = context(&cell);
    context.mem_atomic = denied_write;
    context.x[3] = 9;
    let mut memory = CodeMem(words(8, false));
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    step(&mut jit, &mut context, &mut memory);
    let result = jit.run_bounded(&mut context, &mut memory, 2).unwrap();
    let StopReason::MemoryFault(fault) = result.stop else {
        panic!("expected write memory fault");
    };
    assert_eq!(fault.pc, CODE + 4);
    assert_eq!(context.mem_fault_addr, DATA);
    assert_eq!(context.mem_fault_size, 8);
    assert_eq!(context.mem_fault_access, 1);
    assert_eq!(context.exclusive_size, 0);
    assert_eq!(cell.load(Ordering::SeqCst), 7);
}

#[test]
fn faulting_exclusive_load_does_not_leave_a_reservation() {
    let cell = AtomicU64::new(7);
    let mut context = context(&cell);
    context.mem_read = denied_read;
    let mut memory = CodeMem(words(8, false));
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    let result = jit.run_bounded(&mut context, &mut memory, 1).unwrap();
    assert!(matches!(result.stop, StopReason::MemoryFault(_)));
    assert_eq!(context.pc, CODE);
    assert_eq!(context.mem_fault_addr, DATA);
    assert_eq!(context.mem_fault_size, 8);
    assert_eq!(context.exclusive_size, 0);
}

// These tests establish atomic value-CAS behavior, not full global-monitor
// parity. Same-value competing writes and ABA require a shared generation or
// reservation monitor; a value-only callback cannot distinguish them.
