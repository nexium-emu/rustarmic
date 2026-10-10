use rustarmic::{CpuContext, FlatMemory, Jit, JitConfig, StopReason};

const CODE: u64 = 0x1000;
const DATA: u64 = 0x10_0000;
const CANARY: u64 = 0xfeed_face_1234_5678;

unsafe extern "C" fn read(context: *mut CpuContext, address: u64, size: u8) {
    let ctx = unsafe { &mut *context };
    let count = (ctx.io_value[1] & 0xffff) + 1;
    ctx.io_value[1] = count | ((size as u64) << 16) | (address.wrapping_sub(DATA) << 32);
    let offset = address.checked_sub(ctx.mem_base_va);
    if offset
        .and_then(|offset| offset.checked_add(size as u64))
        .is_none_or(|end| end > ctx.mem_size)
    {
        if ctx.mem_fault == 0 {
            ctx.mem_fault = 1;
            ctx.mem_fault_access = 0;
            ctx.mem_fault_cause = 0;
            ctx.mem_fault_addr = address;
            ctx.mem_fault_size = size;
            ctx.mem_fault_pc = ctx.pc;
        }
        ctx.io_value[0] = 0;
        return;
    }
    let mut bytes = [0; 8];
    unsafe {
        bytes[..size as usize].copy_from_slice(std::slice::from_raw_parts(
            ctx.mem_base.add(offset.unwrap() as usize),
            size as usize,
        ));
    }
    ctx.io_value[0] = u64::from_le_bytes(bytes);
}

fn encoding(count: u32, size: u32, q: bool, rn: usize, rt: usize, post: Option<usize>) -> u32 {
    let shape = count - 1;
    let mut word = 0x0d40_c000
        | (u32::from(q) << 30)
        | ((shape & 1) << 21)
        | ((shape >> 1) << 13)
        | (size << 10)
        | ((rn as u32) << 5)
        | rt as u32;
    if let Some(rm) = post {
        word |= 0x0080_0000 | ((rm as u32) << 16);
    }
    word
}

fn vectors() -> [[u64; 2]; 32] {
    std::array::from_fn(|reg| [CANARY ^ reg as u64, !CANARY ^ (reg as u64 * 37)])
}

fn check(count: u32, size: u32, q: bool, rn: usize, rt: usize, post: Option<usize>, end: bool) {
    let word = encoding(count, size, q, rn, rt, post);
    let bytes = 1usize << size;
    let total = count as usize * bytes;
    let offset = if rn == 31 { 32 } else { 33 };
    for use_fastmem in [false, true] {
        let mut data: Vec<u8> = (0..if end { offset + total } else { 128 })
            .map(|byte| (byte * 17 + 0x81) as u8)
            .collect();
        let old_data = data.clone();
        let mut ctx = CpuContext::default();
        ctx.pc = CODE;
        ctx.x.fill(CANARY);
        ctx.x[10] = 0x25;
        ctx.sp = DATA + offset as u64;
        if rn != 31 {
            ctx.x[rn] = DATA + offset as u64;
        }
        ctx.v = vectors();
        ctx.nzcv = 0xa;
        ctx.fpcr = 0x100_0000;
        ctx.fpsr = 0x20;
        ctx.mem_base = data.as_mut_ptr();
        ctx.mem_base_va = DATA;
        ctx.mem_size = data.len() as u64;
        ctx.mem_read = read;
        let old_x = ctx.x;
        let old_sp = ctx.sp;
        let mut expected_v = ctx.v;
        for index in 0..count as usize {
            let element = &data[offset + index * bytes..offset + (index + 1) * bytes];
            let mut output = [0; 16];
            let active = if q { 16 } else { 8 };
            for lane in output[..active].chunks_exact_mut(bytes) {
                lane.copy_from_slice(element);
            }
            expected_v[(rt + index) & 31] = [
                u64::from_le_bytes(output[..8].try_into().unwrap()),
                u64::from_le_bytes(output[8..].try_into().unwrap()),
            ];
        }
        let mut memory = FlatMemory::new(CODE, 8);
        memory.write_u32(CODE, word);
        memory.write_u32(CODE + 4, 0xd420_0000);
        let mut jit = Jit::new(JitConfig {
            use_fastmem,
            ..JitConfig::default()
        })
        .unwrap();
        let outcome = jit.run_bounded(&mut ctx, &mut memory, 2).unwrap();
        assert_eq!(
            outcome.stop,
            StopReason::Brk(0),
            "word={word:#x}, fastmem={use_fastmem}"
        );
        assert_eq!(outcome.retired, 2);
        assert_eq!(ctx.v, expected_v, "word={word:#x}, fastmem={use_fastmem}");
        let mut expected_x = old_x;
        let mut expected_sp = old_sp;
        if let Some(rm) = post {
            let increment = if rm == 31 { total as u64 } else { old_x[rm] };
            if rn == 31 {
                expected_sp = old_sp.wrapping_add(increment);
            } else {
                expected_x[rn] = old_x[rn].wrapping_add(increment);
            }
        }
        assert_eq!(ctx.x, expected_x);
        assert_eq!(ctx.sp, expected_sp);
        assert_eq!(ctx.nzcv, 0xa);
        assert_eq!(ctx.fpcr, 0x100_0000);
        assert_eq!(ctx.fpsr, 0x20);
        assert_eq!(data, old_data);
        let callbacks = if use_fastmem {
            0
        } else {
            count as u64 | ((bytes as u64) << 16) | (((offset + total - bytes) as u64) << 32)
        };
        assert_eq!(
            ctx.io_value[1], callbacks,
            "exact element callbacks for {word:#x}"
        );
    }
}

#[test]
fn all_replicate_shapes_arrangements_addressing_and_vector_wrap() {
    for count in 1..=4 {
        for size in 0..4 {
            for q in [false, true] {
                for rn in [9, 31] {
                    for rt in [0, 30, 31] {
                        for post in [None, Some(31), Some(10)] {
                            check(count, size, q, rn, rt, post, false);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn replicate_writeback_snapshots_aliased_base_and_uses_scalar_footprint() {
    for count in 1..=4 {
        for size in 0..4 {
            for q in [false, true] {
                for rt in [9, 31] {
                    check(count, size, q, 9, rt, Some(9), false);
                }
            }
        }
    }
}

#[test]
fn wonder_ld1r_and_mapping_end_exact_accesses() {
    assert_eq!(encoding(1, 2, true, 8, 2, None), 0x4d40_c902);
    check(1, 2, true, 8, 2, None, false);
    for count in 1..=4 {
        for size in 0..4 {
            for q in [false, true] {
                for rn in [9, 31] {
                    for post in [None, Some(31), Some(10)] {
                        check(count, size, q, rn, 31, post, true);
                    }
                }
            }
        }
    }
}

#[test]
fn reserved_replicate_encodings_retire_nothing_and_preserve_registers() {
    for count in 1..=4 {
        for size in 0..4 {
            for reserved in [1 << 12, 1 << 16] {
                let mut memory = FlatMemory::new(CODE, 4);
                memory.write_u32(CODE, encoding(count, size, true, 9, 31, None) | reserved);
                let mut ctx = CpuContext::default();
                ctx.pc = CODE;
                ctx.x.fill(CANARY);
                ctx.v = vectors();
                let mut jit = Jit::new(JitConfig::default()).unwrap();
                let outcome = jit.run_bounded(&mut ctx, &mut memory, 1).unwrap();
                assert!(matches!(outcome.stop, StopReason::Unsupported(_)));
                assert_eq!(outcome.retired, 0);
                assert_eq!(ctx.x, [CANARY; 31]);
                assert_eq!(ctx.v, vectors());
            }
        }
    }
}
