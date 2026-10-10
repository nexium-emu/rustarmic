use rustarmic::{CpuContext, FlatMemory, Jit, JitConfig, StopReason};

const CODE: u64 = 0x1000;
const DATA: u64 = 0x10_0000;
const GUARD: u64 = 0xfeed_face_1234_5678;

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
    let mut bytes = [0u8; 8];
    unsafe {
        bytes[..size as usize].copy_from_slice(std::slice::from_raw_parts(
            ctx.mem_base.add(offset.unwrap() as usize),
            size as usize,
        ));
    }
    ctx.io_value[0] = u64::from_le_bytes(bytes);
}

#[derive(Clone, Copy)]
struct Shape {
    opcode: u32,
    count: usize,
    interleaved: bool,
}
const SHAPES: [Shape; 7] = [
    Shape {
        opcode: 7,
        count: 1,
        interleaved: false,
    },
    Shape {
        opcode: 10,
        count: 2,
        interleaved: false,
    },
    Shape {
        opcode: 6,
        count: 3,
        interleaved: false,
    },
    Shape {
        opcode: 2,
        count: 4,
        interleaved: false,
    },
    Shape {
        opcode: 8,
        count: 2,
        interleaved: true,
    },
    Shape {
        opcode: 4,
        count: 3,
        interleaved: true,
    },
    Shape {
        opcode: 0,
        count: 4,
        interleaved: true,
    },
];
fn encoding(shape: Shape, size: u32, q: bool, rn: usize, rt: usize, post: Option<usize>) -> u32 {
    let mut word = 0x0c40_0000
        | (u32::from(q) << 30)
        | (shape.opcode << 12)
        | (size << 10)
        | ((rn as u32) << 5)
        | rt as u32;
    if let Some(rm) = post {
        word |= 0x0080_0000 | ((rm as u32) << 16);
    }
    word
}
fn vector_bytes(reg: usize) -> [u8; 16] {
    std::array::from_fn(|byte| (reg * 37 + byte * 11 + 3) as u8)
}
fn initial_vectors() -> [[u64; 2]; 32] {
    std::array::from_fn(|reg| {
        let bytes = vector_bytes(reg);
        [
            u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..].try_into().unwrap()),
        ]
    })
}

// Invert the memory byte index independently of the emitter's nested loops.
fn oracle(shape: Shape, size: u32, q: bool, rt: usize, data: &[u8]) -> [[u64; 2]; 32] {
    let element = 1usize << size;
    let width = if q { 16 } else { 8 };
    let mut vectors = initial_vectors();
    let mut output = vec![[0u8; 16]; shape.count];
    for (offset, byte) in data.iter().copied().enumerate() {
        let (reg, vector_byte) = if shape.interleaved {
            (
                (offset / element) % shape.count,
                (offset / (element * shape.count)) * element + offset % element,
            )
        } else {
            (offset / width, offset % width)
        };
        output[reg][vector_byte] = byte;
    }
    for (index, bytes) in output.iter().enumerate() {
        vectors[(rt + index) & 31] = [
            u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..].try_into().unwrap()),
        ];
    }
    vectors
}

fn check(
    shape: Shape,
    size: u32,
    q: bool,
    rn: usize,
    rt: usize,
    post: Option<usize>,
    exact_end: bool,
) {
    let word = encoding(shape, size, q, rn, rt, post);
    let width = if q { 16 } else { 8 };
    let total = width * shape.count;
    let element = 1usize << size;
    let offset = if rn == 31 { 32 } else { 33 };
    for use_fastmem in [false, true] {
        let mut data: Vec<u8> = (0..if exact_end { offset + total } else { 160 })
            .map(|byte| (byte * 17 + 0x81) as u8)
            .collect();
        let original_data = data.clone();
        let mut ctx = CpuContext::default();
        ctx.pc = CODE;
        ctx.x.fill(GUARD);
        ctx.x[10] = 0x25;
        ctx.sp = DATA + offset as u64;
        if rn != 31 {
            ctx.x[rn] = DATA + offset as u64;
        }
        ctx.v = initial_vectors();
        ctx.nzcv = 0xa;
        ctx.fpcr = 0x100_0000;
        ctx.fpsr = 0x20;
        ctx.mem_base = data.as_mut_ptr();
        ctx.mem_base_va = DATA;
        ctx.mem_size = data.len() as u64;
        ctx.mem_read = read;
        let old_x = ctx.x;
        let old_sp = ctx.sp;
        let expected_v = oracle(shape, size, q, rt, &data[offset..offset + total]);
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
            "word={word:#x} fastmem={use_fastmem}"
        );
        assert_eq!(outcome.retired, 2);
        assert_eq!(ctx.v, expected_v, "word={word:#x} fastmem={use_fastmem}");
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
        assert_eq!(data, original_data);
        let expected_callbacks = if use_fastmem {
            0
        } else {
            (total / element) as u64
                | ((element as u64) << 16)
                | (((offset + total - element) as u64) << 32)
        };
        assert_eq!(
            ctx.io_value[1], expected_callbacks,
            "exact element widths and addresses"
        );
    }
}

#[test]
fn all_multiple_load_shapes_arrangements_modes_and_wrap() {
    for shape in SHAPES {
        for size in 0..4 {
            for q in [false, true] {
                if shape.interleaved && size == 3 && !q {
                    continue;
                }
                for rn in [9, 31] {
                    for rt in [0, 30, 31] {
                        for post in [None, Some(31), Some(10)] {
                            check(shape, size, q, rn, rt, post, false);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn post_index_same_base_register_and_matching_vector_number() {
    for shape in SHAPES {
        for size in 0..4 {
            for q in [false, true] {
                if shape.interleaved && size == 3 && !q {
                    continue;
                }
                for rt in [9, 31] {
                    check(shape, size, q, 9, rt, Some(9), false);
                }
            }
        }
    }
}

#[test]
fn exact_wonder_ld2_and_accesses_at_mapping_end() {
    assert_eq!(encoding(SHAPES[4], 2, false, 21, 0, None), 0x0c40_8aa0);
    check(SHAPES[4], 2, false, 21, 0, None, false);
    check(SHAPES[4], 2, false, 22, 2, None, false);
    check(SHAPES[4], 2, false, 8, 5, None, false);
    for shape in SHAPES {
        for size in 0..4 {
            for q in [false, true] {
                if shape.interleaved && size == 3 && !q {
                    continue;
                }
                for rn in [9, 31] {
                    check(shape, size, q, rn, 31, Some(31), true);
                }
            }
        }
    }
}

#[test]
fn reserved_interleaved_1d_encodings_leave_registers_unchanged() {
    for shape in SHAPES.into_iter().filter(|shape| shape.interleaved) {
        for post in [None, Some(31)] {
            let mut memory = FlatMemory::new(CODE, 4);
            memory.write_u32(CODE, encoding(shape, 3, false, 9, 31, post));
            let mut ctx = CpuContext::default();
            ctx.pc = CODE;
            ctx.x.fill(GUARD);
            ctx.v = initial_vectors();
            let expected_x = ctx.x;
            let expected_v = ctx.v;
            let mut jit = Jit::new(JitConfig::default()).unwrap();
            let outcome = jit.run_bounded(&mut ctx, &mut memory, 1).unwrap();
            assert!(matches!(outcome.stop, StopReason::Unsupported(_)));
            assert_eq!(outcome.retired, 0);
            assert_eq!(ctx.x, expected_x);
            assert_eq!(ctx.v, expected_v);
        }
    }
}

#[test]
fn first_read_fault_reports_exact_access_and_preserves_memory_bounds() {
    for use_fastmem in [false, true] {
        for size in 0..4 {
            let mut data = [0x5au8; 8];
            let mut ctx = CpuContext::default();
            ctx.pc = CODE;
            ctx.x.fill(GUARD);
            ctx.x[9] = DATA + 8;
            ctx.v = initial_vectors();
            ctx.mem_base = data.as_mut_ptr();
            ctx.mem_base_va = DATA;
            ctx.mem_size = data.len() as u64;
            ctx.mem_read = read;
            let mut memory = FlatMemory::new(CODE, 8);
            memory.write_u32(CODE, encoding(SHAPES[4], size, true, 9, 31, Some(31)));
            memory.write_u32(CODE + 4, 0xd420_0000);
            let mut jit = Jit::new(JitConfig {
                use_fastmem,
                ..JitConfig::default()
            })
            .unwrap();
            let outcome = jit.run_bounded(&mut ctx, &mut memory, 2).unwrap();
            let StopReason::MemoryFault(fault) = outcome.stop else {
                panic!("expected memory fault");
            };
            assert_eq!(fault.pc, CODE);
            assert_eq!(fault.address, DATA + 8);
            assert_eq!(fault.size, (1 << size) as u8);
            // Existing backend limitation: memory faults are checked after
            // the native block, not immediately after each callback. The
            // instruction may therefore modify destinations/writeback before
            // returning this precise fault. No transactional state claim here.
            assert_eq!(data, [0x5a; 8]);
        }
    }
}
