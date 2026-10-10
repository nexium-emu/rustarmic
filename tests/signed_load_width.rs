use rustarmic::{CpuContext, FlatMemory, Jit, JitConfig, StopReason};

const CODE: u64 = 0x1000;
const DATA: u64 = 0x20_0000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const BRK: u32 = 0xd420_0000;

fn jit(use_fastmem: bool) -> Jit {
    // Each fixture image has fewer than ten guest instructions. Keep the
    // generated code semantics unchanged without hundreds of 256MiB caches.
    Jit::new(JitConfig {
        use_fastmem,
        code_cache_bytes: 1024 * 1024,
        ..JitConfig::default()
    })
    .unwrap()
}

#[derive(Clone, Copy, Debug)]
enum Address {
    Unsigned(u32),
    Unscaled(i32),
    Pre(i32),
    Post(i32),
    Register { option: u32, scaled: bool },
}

fn instruction(bytes: u32, wide: bool, rn: u32, rt: u32, rm: u32, address: Address) -> u32 {
    let width = (bytes.trailing_zeros() << 30) | (u32::from(!wide) << 22);
    let mode = match address {
        Address::Unsigned(offset) => 0x3980_0000 | (offset << 10),
        Address::Unscaled(offset) => 0x3880_0000 | (((offset as u32) & 0x1ff) << 12),
        Address::Pre(offset) => 0x3880_0c00 | (((offset as u32) & 0x1ff) << 12),
        Address::Post(offset) => 0x3880_0400 | (((offset as u32) & 0x1ff) << 12),
        Address::Register { option, scaled } => {
            0x38a0_0800 | (rm << 16) | (option << 13) | (u32::from(scaled) << 12)
        }
    };
    mode | width | (rn << 5) | rt
}

// Integer sign-extension oracle, independent of the instruction's opc field.
fn signed_value(value: u64, bytes: u32, wide: bool) -> u64 {
    let signed = match bytes {
        1 => (value as u8 as i8) as i64,
        2 => (value as u16 as i16) as i64,
        4 => (value as u32 as i32) as i64,
        _ => unreachable!(),
    } as u64;
    if wide {
        signed
    } else {
        signed as u32 as u64
    }
}

unsafe extern "C" fn read(context: *mut CpuContext, address: u64, bytes: u8) {
    let context = unsafe { &mut *context };
    let offset = address.checked_sub(context.mem_base_va);
    if offset.is_none_or(|offset| offset + u64::from(bytes) > context.mem_size) {
        context.mem_fault = 1;
        context.mem_fault_access = 0;
        context.mem_fault_cause = 0;
        context.mem_fault_addr = address;
        context.mem_fault_size = bytes;
        context.mem_fault_pc = context.pc;
        context.io_value = [0, 0];
        return;
    }
    let mut value = [0xa5; 8];
    unsafe {
        value[..bytes as usize].copy_from_slice(std::slice::from_raw_parts(
            context.mem_base.add(offset.unwrap() as usize),
            bytes as usize,
        ));
    }
    context.io_value = [
        u64::from_le_bytes(value),
        address | (u64::from(bytes) << 56),
    ];
}

fn context(data: &mut [u8], base: u64) -> CpuContext {
    let mut context = CpuContext::default();
    context.pc = CODE;
    context.x.fill(GUARD);
    context.sp = GUARD;
    context.nzcv = 0xb;
    for (index, value) in context.v.iter_mut().enumerate() {
        *value = [GUARD ^ index as u64, !GUARD];
    }
    context.mem_base = data.as_mut_ptr();
    context.mem_base_va = base;
    context.mem_size = data.len() as u64;
    context.mem_read = read;
    context
}

fn code(base: u64, words: &[u32]) -> FlatMemory {
    let mut memory = FlatMemory::new(base, words.len() * 4);
    for (index, word) in words.iter().enumerate() {
        memory.write_u32(base + index as u64 * 4, *word);
    }
    memory
}

fn samples(bytes: u32) -> Vec<u64> {
    if bytes == 1 {
        return (0..=255).collect();
    }
    let mut values = if bytes == 2 {
        vec![0, 1, 0x7fff, 0x8000, 0x8001, 0xff00, 0xffff]
    } else {
        vec![
            0,
            1,
            0x7fff_ffff,
            0x8000_0000,
            0x8000_0001,
            0xffff_ff00,
            0xffff_ffff,
        ]
    };
    let mut random = 0x8f73_d411_abcdef01u64;
    for _ in 0..32 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        values.push(random);
    }
    values
}

fn check_address_modes(addresses: &[Address]) {
    for &address in addresses {
        for bytes in [1, 2, 4] {
            for wide in [false, true] {
                // LDRSW has only the X destination form.
                if bytes == 4 && !wide {
                    continue;
                }
                for (rn, rt, rm) in [
                    (0, 1, 9),
                    (23, 23, 9),
                    (31, 30, 9),
                    (0, 31, 9),
                    (0, 9, 9),
                    (0, 1, 31),
                ] {
                    if rn == rt && matches!(address, Address::Pre(_) | Address::Post(_)) {
                        continue; // WBOVERLAPLD is constrained-unpredictable.
                    }
                    let word = instruction(bytes, wide, rn, rt, rm, address);
                    for use_fastmem in [false, true] {
                        let mut memory = code(CODE, &[word, BRK]);
                        let mut jit = jit(use_fastmem);
                        // End backing exactly at the transfer's final byte.
                        let mut data = vec![0xa5; 64 + bytes as usize];
                        for value in samples(bytes) {
                            data[64..].copy_from_slice(&value.to_le_bytes()[..bytes as usize]);
                            let expected_data = data.clone();
                            let mut context = context(&mut data, DATA);
                            let offset = match address {
                                Address::Unsigned(offset) => i64::from(offset * bytes),
                                Address::Unscaled(offset)
                                | Address::Pre(offset)
                                | Address::Post(offset) => i64::from(offset),
                                Address::Register { option, scaled } => {
                                    if rm != 31 {
                                        context.x[rm as usize] = match option {
                                            2 => 0xa5a5_5a5a_0000_0003,
                                            3 => 3,
                                            6 => 0x5a5a_a5a5_ffff_fffd,
                                            7 => (-3i64) as u64,
                                            _ => unreachable!(),
                                        };
                                    }
                                    let value = if rm == 31 {
                                        0
                                    } else if option < 4 {
                                        3
                                    } else {
                                        -3
                                    };
                                    value * if scaled { i64::from(bytes) } else { 1 }
                                }
                            };
                            let target = DATA + 64;
                            let base = if matches!(address, Address::Post(_)) {
                                target
                            } else {
                                target.wrapping_sub(offset as u64)
                            };
                            if rn == 31 {
                                context.sp = base;
                            } else {
                                context.x[rn as usize] = base;
                            }
                            let original_v = context.v;
                            let mut expected_x = context.x;
                            let mut expected_sp = context.sp;
                            if rt != 31 {
                                expected_x[rt as usize] = signed_value(value, bytes, wide);
                            }
                            if matches!(address, Address::Pre(_) | Address::Post(_)) {
                                let next = base.wrapping_add(offset as u64);
                                if rn == 31 {
                                    expected_sp = next;
                                } else {
                                    expected_x[rn as usize] = next;
                                }
                            }
                            let outcome = jit.run_bounded(&mut context, &mut memory, 2).unwrap();
                            assert_eq!(
                                outcome.stop,
                                StopReason::Brk(0),
                                "{word:#010x} {address:?} fastmem={use_fastmem}"
                            );
                            assert_eq!(outcome.retired, 2);
                            assert_eq!(
                                context.x, expected_x,
                                "{word:#010x} {address:?} input={value:#x} fastmem={use_fastmem}"
                            );
                            assert_eq!(context.sp, expected_sp);
                            assert_eq!(context.v, original_v);
                            assert_eq!(context.nzcv, 0xb);
                            assert_eq!(data, expected_data);
                            assert_eq!(context.mem_fault, 0);
                            assert_eq!(
                                context.io_value[1],
                                if use_fastmem {
                                    0
                                } else {
                                    target | (u64::from(bytes) << 56)
                                },
                                "callback width/address word={word:#010x} rt={rt} fastmem={use_fastmem}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn signed_loads_unsigned_and_unscaled_offsets_have_correct_destination_width() {
    check_address_modes(&[
        Address::Unsigned(0),
        Address::Unsigned(4095),
        Address::Unscaled(-256),
        Address::Unscaled(-1),
        Address::Unscaled(0),
        Address::Unscaled(255),
    ]);
}

#[test]
fn signed_loads_pre_and_post_index_preserve_exact_writeback() {
    check_address_modes(&[
        Address::Pre(-256),
        Address::Pre(-1),
        Address::Pre(0),
        Address::Pre(255),
        Address::Post(-256),
        Address::Post(-1),
        Address::Post(0),
        Address::Post(255),
    ]);
}

#[test]
fn signed_loads_all_register_extensions_scales_and_source_aliases() {
    let mut addresses = Vec::new();
    for option in [2, 3, 6, 7] {
        for scaled in [false, true] {
            addresses.push(Address::Register { option, scaled });
        }
    }
    check_address_modes(&addresses);
}

#[test]
fn exact_wonder_signed_byte_table_lookup_uses_minus_one_not_positive_u32() {
    const ACTUAL: u64 = 0x820_52dc;
    const GLOBALS: u64 = 0xb7d_3800;
    let words = [
        0x3980_02ca,
        0xd001_ae69,
        0x9120_c129,
        0xd001_ae6b,
        0x9121_016b,
        0xf86a_7929,
        0xf86a_796a,
        0x9b08_7d48,
        BRK,
    ];
    assert_eq!(
        instruction(1, true, 22, 10, 0, Address::Unsigned(0)),
        words[0]
    );
    assert_eq!(
        0xb7d_3830u64.wrapping_add((u32::MAX as u64) << 3),
        0x80b7_d3828
    );
    for use_fastmem in [false, true] {
        let mut data = [0xa5; 0x100];
        data[0x28..0x30].copy_from_slice(&0x1234_5678_9abc_def0u64.to_le_bytes());
        data[0x38..0x40].copy_from_slice(&0xe40u64.to_le_bytes());
        data[0x90] = 0xff;
        let expected_data = data;
        let mut context = context(&mut data, GLOBALS);
        context.pc = ACTUAL;
        context.x[22] = GLOBALS + 0x90;
        context.x[8] = 3;
        let mut expected_x = context.x;
        expected_x[8] = 0x2ac0;
        expected_x[9] = 0x1234_5678_9abc_def0;
        expected_x[10] = 0xe40;
        expected_x[11] = GLOBALS + 0x40;
        let original_v = context.v;
        let mut memory = code(ACTUAL, &words);
        let mut jit = jit(use_fastmem);
        let outcome = jit
            .run_bounded(&mut context, &mut memory, words.len() as u64)
            .unwrap();
        assert_eq!(outcome.stop, StopReason::Brk(0));
        assert_eq!(outcome.retired, words.len() as u64);
        assert_eq!(context.x, expected_x);
        assert_eq!(context.v, original_v);
        assert_eq!(context.sp, GUARD);
        assert_eq!(context.nzcv, 0xb);
        assert_eq!(data, expected_data);
        assert_eq!(context.mem_fault, 0);
        assert_eq!(
            context.io_value[1],
            if use_fastmem {
                0
            } else {
                GLOBALS + 0x38 | (8 << 56)
            }
        );
    }
}

#[test]
fn signed_word_literal_and_pair_offset_pre_post_remain_sign_extended() {
    for use_fastmem in [false, true] {
        for rt in [1, 31] {
            let mut data = 0x8000_0001u32.to_le_bytes();
            let mut context = context(&mut data, CODE + 0x100);
            let mut expected_x = context.x;
            if rt != 31 {
                expected_x[rt as usize] = 0xffff_ffff_8000_0001;
            }
            let original_v = context.v;
            let mut memory = code(CODE, &[0x9800_0800 | rt, BRK]);
            let mut jit = jit(use_fastmem);
            assert_eq!(
                jit.run_bounded(&mut context, &mut memory, 2).unwrap().stop,
                StopReason::Brk(0)
            );
            assert_eq!(context.x, expected_x);
            assert_eq!(context.v, original_v);
            assert_eq!(context.sp, GUARD);
            assert_eq!(context.nzcv, 0xb);
        }
        for (pair_base, writeback, post) in [
            (0x6940_0000, false, false),
            (0x69c0_0000, true, false),
            (0x68c0_0000, true, true),
        ] {
            for offset in [-256i32, 0, 252] {
                for (rn, rt, rt2) in [(0, 1, 2), (31, 30, 29), (0, 31, 30)] {
                    let mut data = [0xa5; 72];
                    data[64..68].copy_from_slice(&0x8000_0001u32.to_le_bytes());
                    data[68..72].copy_from_slice(&0x7fff_ffffu32.to_le_bytes());
                    let expected_data = data;
                    let mut context = context(&mut data, DATA);
                    let target = DATA + 64;
                    let base = if post {
                        target
                    } else {
                        target.wrapping_sub(offset as u64)
                    };
                    if rn == 31 {
                        context.sp = base;
                    } else {
                        context.x[rn as usize] = base;
                    }
                    let mut expected_x = context.x;
                    let mut expected_sp = context.sp;
                    if rt != 31 {
                        expected_x[rt as usize] = 0xffff_ffff_8000_0001;
                    }
                    expected_x[rt2 as usize] = 0x7fff_ffff;
                    if writeback {
                        let next = base.wrapping_add(offset as u64);
                        if rn == 31 {
                            expected_sp = next;
                        } else {
                            expected_x[rn as usize] = next;
                        }
                    }
                    let original_v = context.v;
                    let word = pair_base
                        | (((offset / 4) as u32 & 0x7f) << 15)
                        | (rt2 << 10)
                        | (rn << 5)
                        | rt;
                    let mut memory = code(CODE, &[word, BRK]);
                    let mut jit = jit(use_fastmem);
                    let outcome = jit.run_bounded(&mut context, &mut memory, 2).unwrap();
                    assert_eq!(outcome.stop, StopReason::Brk(0));
                    assert_eq!(context.x, expected_x, "{word:#010x}");
                    assert_eq!(context.sp, expected_sp);
                    assert_eq!(context.v, original_v);
                    assert_eq!(context.nzcv, 0xb);
                    assert_eq!(data, expected_data);
                }
            }
        }
    }
}

#[test]
fn reserved_register_offset_extensions_stop_without_retiring_or_reading() {
    for bytes in [1, 2] {
        for wide in [false, true] {
            for option in [0, 1, 4, 5] {
                let word = instruction(
                    bytes,
                    wide,
                    0,
                    1,
                    9,
                    Address::Register {
                        option,
                        scaled: false,
                    },
                );
                let mut data = [0xff; 8];
                let mut context = context(&mut data, DATA);
                context.x[0] = DATA;
                let original_x = context.x;
                let original_v = context.v;
                let mut memory = code(CODE, &[word]);
                let mut jit = jit(false);
                let outcome = jit.run_bounded(&mut context, &mut memory, 1).unwrap();
                assert!(
                    matches!(outcome.stop, StopReason::Unsupported(info) if info.opcode == word && info.pc == CODE)
                );
                assert_eq!(outcome.retired, 0);
                assert_eq!(context.x, original_x);
                assert_eq!(context.v, original_v);
                assert_eq!(context.sp, GUARD);
                assert_eq!(context.nzcv, 0xb);
                assert_eq!(context.io_value, [0, 0]);
                assert_eq!(data, [0xff; 8]);
            }
        }
    }
}

#[test]
fn discarded_signed_loads_still_report_unmapped_reads_in_every_address_mode() {
    for address in [
        Address::Unsigned(0),
        Address::Unscaled(0),
        Address::Pre(0),
        Address::Post(0),
        Address::Register {
            option: 3,
            scaled: false,
        },
    ] {
        for bytes in [1, 2, 4] {
            for wide in [false, true] {
                if bytes == 4 && !wide {
                    continue;
                }
                let word = instruction(bytes, wide, 0, 31, 9, address);
                for use_fastmem in [false, true] {
                    let mut data = [0xa5; 8];
                    let mut context = context(&mut data, DATA);
                    context.x[0] = DATA + 16;
                    context.x[9] = 0;
                    let original_x = context.x;
                    let original_v = context.v;
                    let mut memory = code(CODE, &[word, BRK]);
                    let mut jit = jit(use_fastmem);
                    let outcome = jit.run_bounded(&mut context, &mut memory, 2).unwrap();
                    let StopReason::MemoryFault(fault) = outcome.stop else {
                        panic!(
                            "discarded read {word:#010x} did not fault: {:?}",
                            outcome.stop
                        );
                    };
                    assert_eq!(
                        (fault.pc, fault.address, fault.size),
                        (CODE, DATA + 16, bytes as u8)
                    );
                    assert_eq!(fault.access, rustarmic::MemoryAccess::Read);
                    assert_eq!(context.x, original_x);
                    assert_eq!(context.v, original_v);
                    assert_eq!(context.sp, GUARD);
                    assert_eq!(context.nzcv, 0xb);
                    assert_eq!(data, [0xa5; 8]);
                }
            }
        }
    }
}

#[test]
fn discarded_unsigned_qword_and_overwritten_signed_result_still_read_and_fault() {
    for (word, bytes, overwrite) in [
        (0xf940_001f, 8, false), // LDR XZR,[X0].
        (0x3980_0001, 1, true),  // LDRSB X1,[X0]; MOV X1,#1.
    ] {
        for use_fastmem in [false, true] {
            for mapped in [false, true] {
                let mut data = [0xff; 8];
                let mut context = context(&mut data, DATA);
                let address = DATA + if mapped { 0 } else { 16 };
                context.x[0] = address;
                let mut expected_x = context.x;
                if overwrite {
                    expected_x[1] = 1;
                }
                let original_v = context.v;
                let mut words = vec![word];
                if overwrite {
                    words.push(0xd280_0021);
                }
                words.push(BRK);
                let mut memory = code(CODE, &words);
                let mut jit = jit(use_fastmem);
                let outcome = jit
                    .run_bounded(&mut context, &mut memory, words.len() as u64)
                    .unwrap();
                if mapped {
                    assert_eq!(outcome.stop, StopReason::Brk(0));
                    assert_eq!(
                        context.io_value[1],
                        if use_fastmem {
                            0
                        } else {
                            address | ((bytes as u64) << 56)
                        }
                    );
                } else {
                    let StopReason::MemoryFault(fault) = outcome.stop else {
                        panic!("lost dead-load fault {word:#010x}");
                    };
                    assert_eq!(
                        (fault.pc, fault.address, fault.size),
                        (CODE, address, bytes)
                    );
                    assert_eq!(fault.access, rustarmic::MemoryAccess::Read);
                }
                // Existing native block-fault model does not roll back later
                // register writes; assert its fault report, not transactionality.
                assert_eq!(context.x, expected_x);
                assert_eq!(context.v, original_v);
                assert_eq!(context.sp, GUARD);
                assert_eq!(context.nzcv, 0xb);
                assert_eq!(data, [0xff; 8]);
            }
        }
    }
}

#[test]
fn optimizer_keeps_all_observable_read_families_and_prior_callback_state() {
    use rustarmic::ir::{Armlet, Block, IrEmitter, Op, Ty};
    for (op, ty) in [
        (Op::Load8, Ty::U8),
        (Op::Load16, Ty::U16),
        (Op::Load32, Ty::U32),
        (Op::Load64, Ty::U64),
        (Op::Load128, Ty::U128),
        (Op::LoadS8, Ty::U64),
        (Op::LoadS16, Ty::U64),
        (Op::LoadS32, Ty::U64),
        (Op::LoadAcq8, Ty::U8),
        (Op::LoadAcq16, Ty::U16),
        (Op::LoadAcq32, Ty::U32),
        (Op::LoadAcq64, Ty::U64),
        (Op::LoadPair8, Ty::U128),
        (Op::LoadPair16, Ty::U128),
        (Op::LoadPair32, Ty::U128),
        (Op::LoadPair64, Ty::U128),
    ] {
        assert!(
            op.has_side_effects(),
            "{op:?} must not be dead-code eliminated"
        );
        assert!(!op.is_pure());
        let mut block = Block::new(CODE);
        let mut emitter = IrEmitter::new(&mut block, CODE);
        let before = emitter.const_u64(1);
        emitter.set_x(0, before);
        let address = emitter.const_u64(DATA);
        emitter.push(Armlet::new(op, ty).with_args(&[address]));
        let after = emitter.const_u64(2);
        emitter.set_x(0, after);
        rustarmic::optimizer::optimize(&mut block);
        let live: Vec<_> = block.iter_live().map(|(_, armlet)| armlet.op).collect();
        assert!(live.contains(&op), "unused {op:?} must remain observable");
        assert_eq!(
            live.iter().filter(|&&op| op == Op::SetX).count(),
            2,
            "read callback may inspect the pre-read context state"
        );
    }
}
