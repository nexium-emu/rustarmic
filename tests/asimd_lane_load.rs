use rustarmic::{CpuContext, FlatMemory, Jit, JitConfig, StopReason};

const CODE_BASE: u64 = 0x1000;
const DATA_BASE: u64 = 0x10_0000;
const CANARY: u64 = 0xfeed_face_1234_5678;

unsafe extern "C" fn load(context: *mut CpuContext, address: u64, size: u8) {
    let context = unsafe { &mut *context };
    let offset = (address - context.mem_base_va) as usize;
    let mut value = [0; 8];
    unsafe {
        value[..size as usize].copy_from_slice(std::slice::from_raw_parts(
            context.mem_base.add(offset),
            size as usize,
        ));
    }
    context.io_value[0] = u64::from_le_bytes(value);
    // Only the low io_value word is consumed by these scalar loads. Record
    // callback count, exact width, and address in the unused scratch word.
    let count = (context.io_value[1] & 0xffff) + 1;
    context.io_value[1] = count | ((size as u64) << 16) | ((offset as u64) << 32);
}

fn encoding(bytes: u32, lane: u32, rn: u32, rt: u32, post: Option<u32>) -> u32 {
    let (base, q, s, size) = match bytes {
        1 => (0x0d40_0000, lane >> 3, (lane >> 2) & 1, lane & 3),
        2 => (0x0d40_4000, lane >> 2, (lane >> 1) & 1, (lane & 1) << 1),
        4 => (0x0d40_8000, lane >> 1, lane & 1, 0),
        8 => (0x0d40_8400, lane, 0, 1),
        _ => unreachable!(),
    };
    let mut word = base | (q << 30) | (s << 12) | (size << 10) | (rn << 5) | rt;
    if let Some(rm) = post {
        word |= 0x0080_0000 | (rm << 16);
    }
    word
}

fn vector_bytes(register: usize) -> [u8; 16] {
    std::array::from_fn(|byte| (register * 37 + byte * 11 + 3) as u8)
}

fn check(
    word: u32,
    bytes: usize,
    lane: usize,
    rn: usize,
    rt: usize,
    post: Option<usize>,
    mapping_end: bool,
) {
    for use_fastmem in [false, true] {
        let offset = if rn == 31 { 32 } else { 33 };
        let mut data: Vec<u8> = (0..if mapping_end { offset + bytes } else { 128 })
            .map(|byte| (byte * 17 + 0x81) as u8)
            .collect();
        let original_data = data.clone();
        let mut memory = FlatMemory::new(CODE_BASE, 8);
        memory.write_u32(CODE_BASE, word);
        memory.write_u32(CODE_BASE + 4, 0xd420_0000); // BRK #0.
        let mut context = CpuContext::default();
        context.pc = CODE_BASE;
        context.nzcv = 0xa;
        context.x.fill(CANARY);
        context.x[9] = 0x25;
        context.sp = DATA_BASE + offset as u64;
        if rn != 31 {
            context.x[rn] = DATA_BASE + offset as u64;
        }
        for register in 0..32 {
            let vector = vector_bytes(register);
            context.v[register] = [
                u64::from_le_bytes(vector[..8].try_into().unwrap()),
                u64::from_le_bytes(vector[8..].try_into().unwrap()),
            ];
        }
        context.mem_base = data.as_mut_ptr();
        context.mem_base_va = DATA_BASE;
        context.mem_size = data.len() as u64;
        context.mem_read = load;
        let original_x = context.x;
        let original_sp = context.sp;
        let mut wanted_vectors = context.v;
        let mut vector = vector_bytes(rt);
        vector[lane * bytes..(lane + 1) * bytes].copy_from_slice(&data[offset..offset + bytes]);
        wanted_vectors[rt] = [
            u64::from_le_bytes(vector[..8].try_into().unwrap()),
            u64::from_le_bytes(vector[8..].try_into().unwrap()),
        ];
        let mut jit = Jit::new(JitConfig {
            use_fastmem,
            ..JitConfig::default()
        })
        .unwrap();
        let outcome = jit.run_bounded(&mut context, &mut memory, 2).unwrap();
        assert_eq!(
            outcome.stop,
            StopReason::Brk(0),
            "opcode {word:#010x} fastmem={use_fastmem}"
        );
        assert_eq!(outcome.retired, 2);
        assert_eq!(
            context.v, wanted_vectors,
            "opcode {word:#010x} fastmem={use_fastmem}: only selected lane may change"
        );
        let mut wanted_x = original_x;
        let mut wanted_sp = original_sp;
        if let Some(rm) = post {
            let increment = if rm == 31 {
                bytes as u64
            } else {
                original_x[rm]
            };
            if rn == 31 {
                wanted_sp = original_sp.wrapping_add(increment);
            } else {
                wanted_x[rn] = original_x[rn].wrapping_add(increment);
            }
        }
        assert_eq!(
            context.x, wanted_x,
            "opcode {word:#010x}: register writeback and GPR preservation"
        );
        assert_eq!(context.sp, wanted_sp, "opcode {word:#010x}: SP writeback");
        assert_eq!(context.nzcv, 0xa);
        assert_eq!(data, original_data, "load must not modify any memory bytes");
        let callback_record = if use_fastmem {
            0
        } else {
            1 | ((bytes as u64) << 16) | ((offset as u64) << 32)
        };
        assert_eq!(
            context.io_value[1], callback_record,
            "exactly one exact-width callback, or bounded fastmem"
        );
    }
}

#[test]
fn wonder_ld1_single_lane_zero() {
    assert_eq!(encoding(4, 0, 16, 2, None), 0x0d40_8202);
    check(0x0d40_8202, 4, 0, 16, 2, None, false);
}

#[test]
fn ld1_every_lane_width_and_addressing_form_preserves_other_lanes() {
    for bytes in [1, 2, 4, 8] {
        for lane in 0..16 / bytes {
            for rn in [16, 31] {
                for rt in [0, 2, 31] {
                    for post in [None, Some(31), Some(9)] {
                        check(
                            encoding(bytes, lane, rn, rt, post),
                            bytes as usize,
                            lane as usize,
                            rn as usize,
                            rt as usize,
                            post.map(|register| register as usize),
                            false,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn ld1_register_post_index_captures_original_base_even_with_vector_alias_number() {
    for bytes in [1, 2, 4, 8] {
        for lane in [0, 16 / bytes - 1] {
            for rn in [8, 16] {
                for rt in [rn, 31] {
                    check(
                        encoding(bytes, lane, rn, rt, Some(rn)),
                        bytes as usize,
                        lane as usize,
                        rn as usize,
                        rt as usize,
                        Some(rn as usize),
                        false,
                    );
                }
            }
        }
    }
}

#[test]
fn ld1_exact_access_width_at_mapping_end() {
    for bytes in [1, 2, 4, 8] {
        for lane in [0, 16 / bytes - 1] {
            for rn in [16, 31] {
                for post in [None, Some(31), Some(9)] {
                    check(
                        encoding(bytes, lane, rn, 31, post),
                        bytes as usize,
                        lane as usize,
                        rn as usize,
                        31,
                        post.map(|register| register as usize),
                        true,
                    );
                }
            }
        }
    }
}
