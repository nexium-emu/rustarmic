use rustarmic::{CpuContext, FlatMemory, Jit, JitConfig, StopReason};

const CODE: u64 = 0x1000;
const DATA: u64 = 0x20_0000;
const CANARY: u64 = 0xfeed_face_1020_3040;

fn encoding(width: u32, immr: u32, imms: u32, rn: u32, rd: u32) -> u32 {
    let base = if width == 64 {
        0x9340_0000
    } else {
        0x1300_0000
    };
    base | (immr << 16) | (imms << 10) | (rn << 5) | rd
}

// Independent signed-field oracle: extract the specified source field, make
// that field a signed i128, then insert at its architectural destination bit.
fn oracle(value: u64, width: u32, immr: u32, imms: u32) -> u64 {
    let (field_bits, source_bit, destination_bit) = if imms >= immr {
        (imms - immr + 1, immr, 0)
    } else {
        (imms + 1, 0, width - immr)
    };
    let field_mask = (1u128 << field_bits) - 1;
    let field = (u128::from(value) >> source_bit) & field_mask;
    let signed = if field & (1u128 << (field_bits - 1)) != 0 {
        field as i128 - (1i128 << field_bits)
    } else {
        field as i128
    };
    let result = (signed << destination_bit) as u64;
    if width == 32 {
        result & u64::from(u32::MAX)
    } else {
        result
    }
}

fn check(width: u32, immr: u32, imms: u32, rn: usize, rd: usize, values: &[u64]) {
    let word = encoding(width, immr, imms, rn as u32, rd as u32);
    let mut memory = FlatMemory::new(CODE, 8);
    memory.write_u32(CODE, word);
    memory.write_u32(CODE + 4, 0xd420_0000);
    let mut jit = Jit::new(JitConfig {
        code_cache_bytes: 64 * 1024,
        ..JitConfig::default()
    })
    .unwrap();
    for &value in values {
        let mut context = CpuContext::default();
        context.pc = CODE;
        context.x.fill(CANARY);
        context.sp = CANARY ^ 0x1234;
        context.nzcv = 0xd;
        context.v.fill([CANARY, !CANARY]);
        if rn != 31 {
            context.x[rn] = value;
        }
        let original_v = context.v;
        let mut expected = context.x;
        if rd != 31 {
            expected[rd] = oracle(if rn == 31 { 0 } else { value }, width, immr, imms);
        }
        let outcome = jit.run_bounded(&mut context, &mut memory, 2).unwrap();
        assert_eq!(outcome.stop, StopReason::Brk(0));
        assert_eq!(outcome.retired, 2);
        assert_eq!(context.x, expected, "word={word:#010x} value={value:#018x}");
        assert_eq!(context.sp, CANARY ^ 0x1234);
        assert_eq!(context.nzcv, 0xd);
        assert_eq!(context.v, original_v);
    }
}

#[test]
fn exact_wonder_sbfiz_sign_extends_inserted_bit33() {
    assert_eq!(encoding(64, 62, 31, 10, 10), 0x937e_7d4a);
    assert_eq!(oracle(0xffff_ffff, 64, 62, 31), (-4i64) as u64);
    assert_eq!(oracle(0x8000_0000, 64, 62, 31), (-0x2_0000_0000i64) as u64);
    check(
        64,
        62,
        31,
        10,
        10,
        &[
            0,
            1,
            0x7fff_ffff,
            0x8000_0000,
            0xffff_ffff,
            0xa5a5_a5a5_ffff_fffe,
        ],
    );
}

#[test]
fn every_legal_signed_bitfield_wrap_and_extract_matches_integer_oracle() {
    for width in [32, 64] {
        for immr in 0..width {
            for imms in 0..width {
                check(
                    width,
                    immr,
                    imms,
                    10,
                    10,
                    &[
                        0,
                        u64::MAX,
                        0x0123_4567_89ab_cdef,
                        0xfedc_ba98_7654_3210,
                        1u64 << imms,
                        (1u64 << imms).wrapping_sub(1),
                    ],
                );
            }
        }
    }
}

#[test]
fn signed_bitfield_preserves_other_registers_and_handles_zero_aliases() {
    for width in [32, 64] {
        for (rn, rd) in [(0, 10), (10, 0), (31, 10), (10, 31), (31, 31)] {
            for (immr, imms) in [(width - 1, 0), (width - 2, 15), (0, width - 1), (8, 15)] {
                check(
                    width,
                    immr,
                    imms,
                    rn,
                    rd,
                    &[0, 0x8000_8080_ffff_8001, u64::MAX],
                );
            }
        }
    }
}

unsafe extern "C" fn read(context: *mut CpuContext, address: u64, size: u8) {
    let context = unsafe { &mut *context };
    let offset = address
        .checked_sub(DATA)
        .expect("signed indexed address below fixture");
    assert!(
        offset + u64::from(size) <= context.mem_size,
        "invalid signed index {address:#x}"
    );
    let mut value = [0; 8];
    unsafe {
        value[..size as usize].copy_from_slice(std::slice::from_raw_parts(
            context.mem_base.add(offset as usize),
            size as usize,
        ));
    }
    context.io_value = [u64::from_le_bytes(value), address];
}

#[test]
fn wonder_madd_sbfiz_indexed_load_chain_uses_negative_word_offset() {
    for use_fastmem in [false, true] {
        let mut data = [0xa5; 64];
        data[16..24].copy_from_slice(&(DATA + 48).to_le_bytes());
        data[28..32].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        let original_data = data;
        let mut memory = FlatMemory::new(CODE, 20);
        for (index, word) in [
            0x1b0c_292a, // MADD W10,W9,W12,W10: 2*3 + (-11) = -5.
            0xf940_096b, // LDR X11,[X11,#16]: table base DATA+48.
            0x937e_7d4a, // SBFIZ X10,X10,#2,#32: -5*4 = -20.
            0xb86a_696a, // LDR W10,[X11,X10]: table DATA+28, not +16GiB.
            0xd420_0000,
        ]
        .into_iter()
        .enumerate()
        {
            memory.write_u32(CODE + index as u64 * 4, word);
        }
        let mut context = CpuContext::default();
        context.pc = CODE;
        context.x.fill(CANARY);
        context.x[9] = 2;
        context.x[12] = 3;
        context.x[10] = (-11i32) as u32 as u64;
        context.x[11] = DATA;
        context.sp = CANARY;
        context.nzcv = 0xb;
        context.mem_base = data.as_mut_ptr();
        context.mem_base_va = DATA;
        context.mem_size = data.len() as u64;
        context.mem_read = read;
        let mut expected = context.x;
        expected[10] = 0xdead_beef;
        expected[11] = DATA + 48;
        let mut jit = Jit::new(JitConfig {
            use_fastmem,
            ..JitConfig::default()
        })
        .unwrap();
        let outcome = jit.run_bounded(&mut context, &mut memory, 5).unwrap();
        assert_eq!(outcome.stop, StopReason::Brk(0));
        assert_eq!(outcome.retired, 5);
        assert_eq!(context.x, expected);
        assert_eq!(context.sp, CANARY);
        assert_eq!(context.nzcv, 0xb);
        assert_eq!(data, original_data);
        if !use_fastmem {
            assert_eq!(context.io_value[1], DATA + 28);
        }
    }
}
