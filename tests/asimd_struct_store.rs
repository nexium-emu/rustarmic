use rustarmic::{CpuContext, FlatMemory, Jit, JitConfig, StopReason};

const CODE_BASE: u64 = 0x1000;
const DATA_BASE: u64 = 0x10_0000;
const CANARY: u64 = 0xfeed_face_1234_5678;

unsafe extern "C" fn store(context: *mut CpuContext, address: u64, size: u8) {
    let context = unsafe { &mut *context };
    let offset = (address - context.mem_base_va) as usize;
    let bytes = context.io_value[0].to_le_bytes();
    unsafe {
        std::slice::from_raw_parts_mut(context.mem_base.add(offset), size as usize)
            .copy_from_slice(&bytes[..size as usize]);
    }
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
    let mut opcode = 0x0c00_0000
        | (u32::from(q) << 30)
        | (shape.opcode << 12)
        | (size << 10)
        | ((rn as u32) << 5)
        | rt as u32;
    if let Some(rm) = post {
        opcode |= 0x0080_0000 | ((rm as u32) << 16);
    }
    opcode
}

fn vector_bytes(reg: usize) -> [u8; 16] {
    std::array::from_fn(|byte| (reg * 17 + byte * 37) as u8)
}

fn oracle(shape: Shape, bytes: usize, q: bool, rt: usize) -> Vec<u8> {
    let width = if q { 16 } else { 8 };
    let vectors: Vec<_> = (0..shape.count)
        .map(|reg| vector_bytes((rt + reg) & 31))
        .collect();
    let mut output = Vec::new();
    if shape.interleaved {
        for lane in 0..width / bytes {
            for vector in &vectors {
                output.extend_from_slice(&vector[lane * bytes..(lane + 1) * bytes]);
            }
        }
    } else {
        for vector in &vectors {
            output.extend_from_slice(&vector[..width]);
        }
    }
    output
}

fn check(shape: Shape, size: u32, q: bool, rn: usize, rt: usize, post: Option<usize>) {
    let opcode = encoding(shape, size, q, rn, rt, post);
    for use_fastmem in [false, true] {
        // A general base exercises unaligned element stores; SP stays aligned.
        let data_offset = if rn == 31 { 32 } else { 33 };
        let mut data = [0xa5u8; 160];
        let mut memory = FlatMemory::new(CODE_BASE, 8);
        memory.write_u32(CODE_BASE, opcode);
        memory.write_u32(CODE_BASE + 4, 0xd420_0000);
        let mut context = CpuContext::default();
        context.pc = CODE_BASE;
        context.x.fill(CANARY);
        context.x[10] = 0x25;
        context.sp = DATA_BASE + data_offset as u64;
        if rn != 31 {
            context.x[rn] = DATA_BASE + data_offset as u64;
        }
        for reg in 0..32 {
            let vector = vector_bytes(reg);
            context.v[reg] = [
                u64::from_le_bytes(vector[..8].try_into().unwrap()),
                u64::from_le_bytes(vector[8..].try_into().unwrap()),
            ];
        }
        context.nzcv = 0xa;
        context.mem_base = data.as_mut_ptr();
        context.mem_base_va = DATA_BASE;
        context.mem_size = data.len() as u64;
        context.mem_write = store;
        let original_x = context.x;
        let original_sp = context.sp;
        let original_v = context.v;
        let mut jit = Jit::new(JitConfig {
            use_fastmem,
            ..JitConfig::default()
        })
        .unwrap();
        let outcome = jit.run_bounded(&mut context, &mut memory, 2).unwrap();
        assert_eq!(outcome.stop, StopReason::Brk(0), "opcode {opcode:#010x}");
        assert_eq!(outcome.retired, 2);
        let bytes = oracle(shape, 1 << size, q, rt);
        let mut expected_data = [0xa5u8; 160];
        expected_data[data_offset..data_offset + bytes.len()].copy_from_slice(&bytes);
        assert_eq!(
            data, expected_data,
            "opcode {opcode:#010x} fastmem={use_fastmem}"
        );
        let mut expected_x = original_x;
        let mut expected_sp = original_sp;
        if let Some(rm) = post {
            let increment = if rm == 31 {
                bytes.len() as u64
            } else {
                original_x[rm]
            };
            if rn == 31 {
                expected_sp = original_sp.wrapping_add(increment);
            } else {
                expected_x[rn] = original_x[rn].wrapping_add(increment);
            }
        }
        assert_eq!(context.x, expected_x, "opcode {opcode:#010x}");
        assert_eq!(context.sp, expected_sp, "opcode {opcode:#010x}");
        assert_eq!(
            context.v, original_v,
            "all source registers must be preserved"
        );
        assert_eq!(context.nzcv, 0xa);
    }
}

#[test]
fn all_multiple_store_shapes_and_addressing_forms() {
    for shape in SHAPES {
        for size in 0..4 {
            for q in [false, true] {
                if shape.interleaved && size == 3 && !q {
                    continue;
                }
                for rn in [9, 31] {
                    for rt in [0, 30, 31] {
                        for post in [None, Some(31), Some(10)] {
                            check(shape, size, q, rn, rt, post);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn register_post_index_captures_old_base() {
    for shape in SHAPES {
        for size in 0..4 {
            for q in [false, true] {
                if shape.interleaved && size == 3 && !q {
                    continue;
                }
                check(shape, size, q, 9, 31, Some(9));
            }
        }
    }
}

#[test]
fn exact_wonder_st4_v3_to_v6_4s() {
    let shape = SHAPES[6];
    assert_eq!(encoding(shape, 2, true, 9, 3, None), 0x4c00_0923);
    check(shape, 2, true, 9, 3, None);
}

#[test]
fn reserved_interleaved_1d_formats_are_rejected() {
    for shape in SHAPES.iter().copied().filter(|shape| shape.interleaved) {
        for post in [None, Some(31)] {
            let word = encoding(shape, 3, false, 9, 0, post);
            let mut memory = FlatMemory::new(CODE_BASE, 8);
            memory.write_u32(CODE_BASE, word);
            memory.write_u32(CODE_BASE + 4, 0xd420_0000);
            let mut context = CpuContext::default();
            context.pc = CODE_BASE;
            let mut jit = Jit::new(JitConfig::default()).unwrap();
            let outcome = jit.run_bounded(&mut context, &mut memory, 2).unwrap();
            assert!(
                matches!(outcome.stop, StopReason::Unsupported(_)),
                "reserved opcode {word:#010x}"
            );
            assert_eq!(outcome.retired, 0);
        }
    }
}
