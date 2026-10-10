use rustarmic::{CpuContext, FlatMemory, Jit, JitConfig, StopReason};

const CODE_BASE: u64 = 0x1000;
const DATA_BASE: u64 = 0x10_0000;
const DATA_OFFSET: usize = 32;

unsafe extern "C" fn store(ctx: *mut CpuContext, address: u64, size: u8) {
    let ctx = unsafe { &mut *ctx };
    let offset = (address - ctx.mem_base_va) as usize;
    let bytes = ctx.io_value[0].to_le_bytes();
    unsafe {
        std::slice::from_raw_parts_mut(ctx.mem_base.add(offset), size as usize)
            .copy_from_slice(&bytes[..size as usize]);
    }
}

fn encoding(bytes: u32, lane: u32, rn: u32, rt: u32, post: Option<u32>) -> u32 {
    let (base, q, s, size) = match bytes {
        1 => (0x0d00_0000, lane >> 3, (lane >> 2) & 1, lane & 3),
        2 => (0x0d00_4000, lane >> 2, (lane >> 1) & 1, (lane & 1) << 1),
        4 => (0x0d00_8000, lane >> 1, lane & 1, 0),
        8 => (0x0d00_8400, lane, 0, 1),
        _ => unreachable!(),
    };
    let mut word = base | (q << 30) | (s << 12) | (size << 10) | (rn << 5) | rt;
    if let Some(rm) = post {
        word |= 0x0080_0000 | (rm << 16);
    }
    word
}

fn check_store(word: u32, bytes: usize, lane: usize, rn: usize, rt: usize, post: Option<usize>) {
    for use_fastmem in [false, true] {
        check_store_path(word, bytes, lane, rn, rt, post, use_fastmem);
    }
}

fn check_store_path(
    word: u32,
    bytes: usize,
    lane: usize,
    rn: usize,
    rt: usize,
    post: Option<usize>,
    use_fastmem: bool,
) {
    let mut data = [0xa5u8; 128];
    let mut memory = FlatMemory::new(CODE_BASE, 8);
    memory.write_u32(CODE_BASE, word);
    memory.write_u32(CODE_BASE + 4, 0xd420_0000); // BRK #0
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.nzcv = 0xa;
    ctx.x = [0xfeed_face_1234_5678; 31];
    ctx.x[9] = 0x23;
    ctx.sp = DATA_BASE + DATA_OFFSET as u64;
    if rn != 31 {
        ctx.x[rn] = DATA_BASE + DATA_OFFSET as u64;
    }
    ctx.v[rt] = [0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908];
    ctx.mem_base = data.as_mut_ptr();
    ctx.mem_base_va = DATA_BASE;
    ctx.mem_size = data.len() as u64;
    ctx.mem_write = store;
    let original_x = ctx.x;
    let original_sp = ctx.sp;
    let original_v = ctx.v;
    let mut jit = Jit::new(JitConfig {
        use_fastmem,
        ..JitConfig::default()
    })
    .unwrap();
    let outcome = jit.run_bounded(&mut ctx, &mut memory, 2).unwrap();
    assert_eq!(outcome.stop, StopReason::Brk(0), "opcode {word:#010x}");
    assert_eq!(outcome.retired, 2);
    let mut expected_data = [0xa5u8; 128];
    for i in 0..bytes {
        expected_data[DATA_OFFSET + i] = (lane * bytes + i) as u8;
    }
    assert_eq!(data, expected_data, "opcode {word:#010x}");
    let increment = post.map(|rm| {
        if rm == 31 {
            bytes as u64
        } else {
            original_x[rm]
        }
    });
    let mut expected_x = original_x;
    let mut expected_sp = original_sp;
    if let Some(increment) = increment {
        if rn == 31 {
            expected_sp = original_sp.wrapping_add(increment);
        } else {
            expected_x[rn] = original_x[rn].wrapping_add(increment);
        }
    }
    assert_eq!(ctx.x, expected_x, "opcode {word:#010x}");
    assert_eq!(ctx.sp, expected_sp, "opcode {word:#010x}");
    assert_eq!(ctx.v, original_v, "store must not alter vector registers");
    assert_eq!(ctx.nzcv, 0xa);
}

#[test]
fn wonder_st1_word_lane_two() {
    check_store(0x4d00_8100, 4, 2, 8, 0, None);
}

#[test]
fn st1_every_lane_and_addressing_form() {
    for bytes in [1, 2, 4, 8] {
        for lane in 0..16 / bytes {
            for rn in [8, 31] {
                for post in [None, Some(31), Some(9)] {
                    let word = encoding(bytes, lane, rn, 31, post);
                    check_store(
                        word,
                        bytes as usize,
                        lane as usize,
                        rn as usize,
                        31,
                        post.map(|r| r as usize),
                    );
                }
            }
        }
    }
}

#[test]
fn st1_register_post_index_reads_base_before_writeback() {
    for bytes in [1, 2, 4, 8] {
        let lane = 16 / bytes - 1;
        let word = encoding(bytes, lane, 8, 0, Some(8));
        check_store(word, bytes as usize, lane as usize, 8, 0, Some(8));
    }
}
