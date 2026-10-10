use rustarmic::{CpuContext, FlatMemory, Jit, JitConfig, StopReason};

const CODE: u64 = 0x1000;
const DATA: u64 = 0x10_0000;
const CANARY: u64 = 0xdead_beef_0102_0304;

#[derive(Clone, Copy, Debug)]
enum Address {
    Unsigned(u32),
    Unscaled(i32),
    Pre(i32),
    Post(i32),
    Register { option: u32, scaled: bool },
}

fn opcode(bytes: u32, load: bool, rn: u32, rt: u32, address: Address) -> u32 {
    let width = (bytes.trailing_zeros() << 30) | (u32::from(load) << 22);
    let address = match address {
        Address::Unsigned(offset) => 0x3d00_0000 | (offset << 10),
        Address::Unscaled(offset) => 0x3c00_0000 | (((offset as u32) & 0x1ff) << 12),
        Address::Pre(offset) => 0x3c00_0c00 | (((offset as u32) & 0x1ff) << 12),
        Address::Post(offset) => 0x3c00_0400 | (((offset as u32) & 0x1ff) << 12),
        Address::Register { option, scaled } => {
            0x3c20_0800 | (9 << 16) | (option << 13) | (u32::from(scaled) << 12)
        }
    };
    address | width | (rn << 5) | rt
}

fn set_fault(context: &mut CpuContext, address: u64, size: u8, write: bool) {
    context.mem_fault = 1;
    context.mem_fault_access = u8::from(write);
    context.mem_fault_cause = 0;
    context.mem_fault_addr = address;
    context.mem_fault_size = size;
    context.mem_fault_pc = context.pc;
}

unsafe extern "C" fn read(context: *mut CpuContext, address: u64, size: u8) {
    let context = unsafe { &mut *context };
    let Some(offset) = address.checked_sub(context.mem_base_va) else {
        set_fault(context, address, size, false);
        return;
    };
    if offset
        .checked_add(u64::from(size))
        .is_none_or(|end| end > context.mem_size)
    {
        set_fault(context, address, size, false);
        return;
    }
    // Poison every bit outside the exact-width load result.
    let mut value = [0xa5; 8];
    unsafe {
        value[..size as usize].copy_from_slice(std::slice::from_raw_parts(
            context.mem_base.add(offset as usize),
            size as usize,
        ));
    }
    context.io_value[0] = u64::from_le_bytes(value);
    context.io_value[1] = 1 | (u64::from(size) << 16) | (offset << 32);
}

unsafe extern "C" fn write(context: *mut CpuContext, address: u64, size: u8) {
    let context = unsafe { &mut *context };
    let Some(offset) = address.checked_sub(context.mem_base_va) else {
        set_fault(context, address, size, true);
        return;
    };
    if offset
        .checked_add(u64::from(size))
        .is_none_or(|end| end > context.mem_size)
    {
        set_fault(context, address, size, true);
        return;
    }
    let value = context.io_value[0].to_le_bytes();
    unsafe {
        std::slice::from_raw_parts_mut(context.mem_base.add(offset as usize), size as usize)
            .copy_from_slice(&value[..size as usize]);
    }
    context.io_value[1] = 1 | (u64::from(size) << 16) | (offset << 32);
}

fn registers(context: &mut CpuContext) {
    context.x.fill(CANARY);
    context.sp = CANARY;
    context.nzcv = 0xb;
    for register in 0..32 {
        context.v[register] = [CANARY.wrapping_add(register as u64 * 0x101), !CANARY];
    }
}

fn run_case(bytes: u32, load: bool, rn: u32, rt: u32, address: Address, mapping_end: bool) {
    let word = opcode(bytes, load, rn, rt, address);
    for use_fastmem in [false, true] {
        let target_offset = 64usize;
        let target = DATA + target_offset as u64;
        let mut data: Vec<u8> = (0..if mapping_end {
            target_offset + bytes as usize
        } else {
            128
        })
            .map(|index| (index * 31 + 0x73) as u8)
            .collect();
        let mut expected_data = data.clone();
        let mut context = CpuContext::default();
        registers(&mut context);
        let offset = match address {
            Address::Unsigned(offset) => i64::from(offset * bytes),
            Address::Unscaled(offset) | Address::Pre(offset) | Address::Post(offset) => {
                i64::from(offset)
            }
            Address::Register { option, scaled } => {
                context.x[9] = match option {
                    2 => 0xa5a5_5a5a_0000_0003, // UXTW ignores high 32 bits.
                    3 => 3,
                    6 => 0x5a5a_a5a5_ffff_fffd, // SXTW sign extends low 32 bits.
                    7 => (-3i64) as u64,
                    _ => unreachable!(),
                };
                let value = if option < 4 { 3i64 } else { -3i64 };
                value * if scaled { i64::from(bytes) } else { 1 }
            }
        };
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
        context.pc = CODE;
        context.mem_base = data.as_mut_ptr();
        context.mem_base_va = DATA;
        context.mem_size = data.len() as u64;
        context.mem_read = read;
        context.mem_write = write;
        let mut expected_v = context.v;
        if load {
            let mut value = [0; 8];
            value[..bytes as usize]
                .copy_from_slice(&data[target_offset..target_offset + bytes as usize]);
            expected_v[rt as usize] = [u64::from_le_bytes(value), 0];
        } else {
            let value = context.v[rt as usize][0].to_le_bytes();
            expected_data[target_offset..target_offset + bytes as usize]
                .copy_from_slice(&value[..bytes as usize]);
        }
        let mut expected_x = context.x;
        let mut expected_sp = context.sp;
        if matches!(address, Address::Pre(_) | Address::Post(_)) {
            let after = base.wrapping_add(offset as u64);
            if rn == 31 {
                expected_sp = after;
            } else {
                expected_x[rn as usize] = after;
            }
        }
        let mut memory = FlatMemory::new(CODE, 8);
        memory.write_u32(CODE, word);
        memory.write_u32(CODE + 4, 0xd420_0000);
        let mut jit = Jit::new(JitConfig {
            use_fastmem,
            ..JitConfig::default()
        })
        .unwrap();
        let outcome = jit.run_bounded(&mut context, &mut memory, 2).unwrap();
        assert_eq!(
            outcome.stop,
            StopReason::Brk(0),
            "{word:#010x} {address:?} fastmem={use_fastmem}"
        );
        assert_eq!(outcome.retired, 2);
        assert_eq!(
            context.v, expected_v,
            "vector/source preservation {word:#010x}"
        );
        assert_eq!(
            context.x, expected_x,
            "GPR/writeback preservation {word:#010x}"
        );
        assert_eq!(context.sp, expected_sp);
        assert_eq!(context.nzcv, 0xb);
        assert_eq!(
            data, expected_data,
            "exact width and surrounding canaries {word:#010x}"
        );
        let expected_callback = if use_fastmem {
            0
        } else {
            1 | (u64::from(bytes) << 16) | ((target_offset as u64) << 32)
        };
        assert_eq!(
            context.io_value[1], expected_callback,
            "exact callback width {word:#010x}"
        );
    }
}

#[test]
fn exact_wonder_ldr_b0_x23_offset34() {
    assert_eq!(opcode(1, true, 23, 0, Address::Unsigned(34)), 0x3d40_8ae0);
    run_case(1, true, 23, 0, Address::Unsigned(34), false);
}

#[test]
fn scalar_simd_b_h_all_address_modes_clear_upper_and_preserve_guards() {
    let mut addresses = vec![
        Address::Unsigned(0),
        Address::Unsigned(17),
        Address::Unsigned(4095),
    ];
    for offset in [-256, -1, 0, 255] {
        addresses.extend([
            Address::Unscaled(offset),
            Address::Pre(offset),
            Address::Post(offset),
        ]);
    }
    for option in [2, 3, 6, 7] {
        for scaled in [false, true] {
            addresses.push(Address::Register { option, scaled });
        }
    }
    for bytes in [1, 2] {
        for load in [false, true] {
            // Vn and Xn are separate register banks, even with matching numbers.
            for (rn, rt) in [(23, 0), (23, 23), (31, 31)] {
                for &address in &addresses {
                    run_case(bytes, load, rn, rt, address, false);
                }
            }
        }
    }
}

#[test]
fn exact_access_width_at_mapping_end_and_full_width_fault() {
    for bytes in [1, 2] {
        for load in [false, true] {
            run_case(bytes, load, 31, 31, Address::Unsigned(0), true);
            for use_fastmem in [false, true] {
                let mut data = [0x7du8; 8];
                let mut context = CpuContext::default();
                registers(&mut context);
                context.pc = CODE;
                context.x[23] = DATA + 8;
                context.mem_base = data.as_mut_ptr();
                context.mem_base_va = DATA;
                context.mem_size = 8;
                context.mem_read = read;
                context.mem_write = write;
                let original = data;
                let mut memory = FlatMemory::new(CODE, 8);
                memory.write_u32(CODE, opcode(bytes, load, 23, 0, Address::Unsigned(0)));
                memory.write_u32(CODE + 4, 0xd420_0000);
                let mut jit = Jit::new(JitConfig {
                    use_fastmem,
                    ..JitConfig::default()
                })
                .unwrap();
                let outcome = jit.run_bounded(&mut context, &mut memory, 2).unwrap();
                let StopReason::MemoryFault(fault) = outcome.stop else {
                    panic!("expected exact-width memory fault");
                };
                assert_eq!(fault.pc, CODE);
                assert_eq!(fault.address, DATA + 8);
                assert_eq!(fault.size, bytes as u8);
                assert_eq!(data, original);
            }
        }
    }
}
