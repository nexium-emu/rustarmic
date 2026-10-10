use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const DATA: u64 = 0x20_0000;
const BRK: u32 = 0xd420_0000;
const GUARD: u64 = 0xface_1234_5678_9abc;

struct Code(Vec<u32>);
impl Memory for Code {
    fn fetch_inst(&mut self, address: u64) -> Option<u32> {
        let offset = address.checked_sub(CODE)?;
        if offset & 3 != 0 { return None; }
        self.0.get(offset as usize / 4).copied()
    }
}

fn word(signed: bool, maximum: bool, size: u32, q: bool, rn: u32, rd: u32) -> u32 {
    0x0e30_a800 | (u32::from(!signed) << 29) | (u32::from(!maximum) << 16)
        | (u32::from(q) << 30) | (size << 22) | (rn << 5) | rd
}

fn pack(bytes: [u8; 16]) -> [u64; 2] {
    [u64::from_le_bytes(bytes[..8].try_into().unwrap()),
     u64::from_le_bytes(bytes[8..].try_into().unwrap())]
}

fn oracle(bytes: [u8; 16], signed: bool, maximum: bool, size: u32, q: bool) -> [u64; 2] {
    let width = 1usize << size;
    let lanes = if q { 16 } else { 8 } / width;
    let lane = |index: usize| {
        let slice = &bytes[index * width..(index + 1) * width];
        match (signed, width) {
            (true, 1) => i64::from(slice[0] as i8),
            (true, 2) => i64::from(i16::from_le_bytes(slice.try_into().unwrap())),
            (true, 4) => i64::from(i32::from_le_bytes(slice.try_into().unwrap())),
            (false, 1) => i64::from(slice[0]),
            (false, 2) => i64::from(u16::from_le_bytes(slice.try_into().unwrap())),
            (false, 4) => i64::from(u32::from_le_bytes(slice.try_into().unwrap())),
            _ => unreachable!(),
        }
    };
    let selected = if maximum { (0..lanes).max_by_key(|&i| lane(i)).unwrap() }
                   else { (0..lanes).min_by_key(|&i| lane(i)).unwrap() };
    let mut result = [0; 16];
    result[..width].copy_from_slice(&bytes[selected * width..(selected + 1) * width]);
    pack(result)
}

fn patterns(size: u32) -> Vec<[u8; 16]> {
    let width = 1usize << size;
    let bits = width * 8;
    let sign = 1u64 << (bits - 1);
    let mask = (1u64 << bits) - 1;
    let values = [0, 1, sign - 1, sign, sign + 1, mask - 1, mask, 3];
    let mut result = vec![[0; 16], [0xff; 16]];
    for rotate in 0..8 {
        let mut bytes = [0; 16];
        for (index, lane) in bytes.chunks_exact_mut(width).enumerate() {
            lane.copy_from_slice(&values[(index + rotate) & 7].to_le_bytes()[..width]);
        }
        result.push(bytes);
    }
    for selected in 0..16 / width {
        let mut bytes = [0; 16];
        for (index, lane) in bytes.chunks_exact_mut(width).enumerate() {
            let value = if index == selected { sign } else { 7 };
            lane.copy_from_slice(&value.to_le_bytes()[..width]);
        }
        result.push(bytes);
    }
    result.push(std::array::from_fn(|byte| (byte * 103 + 0xb7) as u8));
    result
}

fn context() -> CpuContext {
    let mut context = CpuContext::default();
    context.pc = CODE;
    context.x.fill(GUARD);
    context.sp = GUARD;
    context.nzcv = 0xb;
    context.fpcr = 0x0340_0000;
    context.fpsr = 0x0800_009f;
    for (index, vector) in context.v.iter_mut().enumerate() {
        *vector = [GUARD ^ index as u64, !GUARD];
    }
    context
}

fn guards(context: &CpuContext, expected_x: [u64; 31]) {
    assert_eq!(context.x, expected_x);
    assert_eq!(context.sp, GUARD);
    assert_eq!(context.nzcv, 0xb);
    assert_eq!(context.fpcr, 0x0340_0000);
    assert_eq!(context.fpsr, 0x0800_009f);
}

#[test]
fn every_legal_integer_reduction_width_kind_alias_and_source_half() {
    for signed in [false, true] {
        for maximum in [false, true] {
            for (size, q) in [(0, false), (0, true), (1, false), (1, true), (2, true)] {
                for (rn, rd) in [(2, 1), (2, 2), (31, 31), (31, 0), (0, 31)] {
                    let instruction = word(signed, maximum, size, q, rn, rd);
                    let mut code = Code(vec![instruction, BRK]);
                    let mut jit = Jit::new(JitConfig::default()).unwrap();
                    for source in patterns(size) {
                        let mut context = context();
                        context.v[rn as usize] = pack(source);
                        let mut expected = context.v;
                        expected[rd as usize] = oracle(source, signed, maximum, size, q);
                        let outcome = jit.run_bounded(&mut context, &mut code, 2).unwrap();
                        assert_eq!(outcome.stop, StopReason::Brk(0), "{instruction:#x}");
                        assert_eq!(outcome.retired, 2);
                        assert_eq!(context.v, expected, "{instruction:#x} {source:02x?}");
                        guards(&context, [GUARD; 31]);
                    }
                }
            }
        }
    }
}

#[test]
fn exact_wonder_sminv_selection_chain_and_signed_table_lookup() {
    assert_eq!(word(true, false, 2, true, 18, 19), 0x4eb1_aa53);
    assert_eq!(word(true, false, 2, true, 1, 2), 0x4eb1_a822);
    let mut code = Code(vec![0x4eb1_aa53, 0x4e04_0673, 0x6eb3_8e53, 0x4e20_1e73,
                             0x4eb1_ba73, 0x1e26_026b, 0x38ab_498b, BRK]);
    let mut jit = Jit::new(JitConfig { use_fastmem: true, ..JitConfig::default() }).unwrap();
    for selected in 1u32..16 {
        let mut context = context();
        let mut source = [0; 16];
        let mut weights = [0; 16];
        for index in 0..4 {
            let value = if selected & (1 << index) != 0 { -7i32 } else { 3 };
            source[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
            weights[index * 4..index * 4 + 4].copy_from_slice(&(1u32 << index).to_le_bytes());
        }
        context.v[18] = pack(source);
        context.v[0] = pack(weights);
        let mut expected = context.v;
        expected[19] = [u64::from(selected), 0];
        let mut table = [0xa5u8; 32];
        table[selected as usize] = 0xfe;
        context.mem_base = table.as_mut_ptr();
        context.mem_base_va = DATA;
        context.mem_size = table.len() as u64;
        context.x[12] = DATA;
        let mut expected_x = context.x;
        expected_x[11] = (-2i64) as u64;
        let outcome = jit.run_bounded(&mut context, &mut code, 8).unwrap();
        assert_eq!(outcome.stop, StopReason::Brk(0));
        assert_eq!(context.v, expected);
        guards(&context, expected_x);
        assert_eq!(table[selected as usize], 0xfe);
        assert_eq!(table.iter().filter(|&&byte| byte == 0xa5).count(), 31);
    }
}

#[test]
fn helper_preserves_live_gpr_values_and_all_other_vectors_under_pressure() {
    let mut words = vec![0x9100_0484, 0x9100_04a5, 0x9100_04c6, 0x9100_04e7];
    let mut expected = context();
    for (index, (signed, maximum, size, q)) in
        [(true, false, 2, true), (false, true, 0, true),
         (true, true, 1, false), (false, false, 1, true)].into_iter().enumerate() {
        let rn = 28 + index as u32;
        let rd = index as u32;
        let input = patterns(size).pop().unwrap();
        expected.v[rn as usize] = pack(input);
        words.push(word(signed, maximum, size, q, rn, rd));
    }
    let mut context = context();
    context.v = expected.v;
    context.x[4..8].copy_from_slice(&[10, 20, 30, 40]);
    for (index, (signed, maximum, size, q)) in
        [(true, false, 2, true), (false, true, 0, true),
         (true, true, 1, false), (false, false, 1, true)].into_iter().enumerate() {
        expected.v[index] = oracle(patterns(size).pop().unwrap(), signed, maximum, size, q);
    }
    words.extend([0x8b05_008a, 0x8b07_00cb, BRK]);
    let mut code = Code(words);
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    let outcome = jit.run_bounded(&mut context, &mut code, 11).unwrap();
    assert_eq!(outcome.stop, StopReason::Brk(0));
    let mut expected_x = [GUARD; 31];
    expected_x[4..8].copy_from_slice(&[11, 21, 31, 41]);
    expected_x[10] = 32;
    expected_x[11] = 72;
    assert_eq!(context.v, expected.v);
    guards(&context, expected_x);
}

#[test]
fn reserved_2s_and_d_forms_do_not_retire_or_mutate_state() {
    for (size, q) in [(2, false), (3, false), (3, true)] {
        let instruction = word(true, false, size, q, 31, 31);
        let mut code = Code(vec![instruction]);
        let mut context = context();
        let original = context.v;
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        let result = jit.run_bounded(&mut context, &mut code, 1).unwrap();
        assert!(matches!(result.stop, StopReason::Unsupported(info) if info.opcode == instruction));
        assert_eq!(result.retired, 0);
        assert_eq!(context.pc, CODE);
        assert_eq!(context.v, original);
        guards(&context, [GUARD; 31]);
    }
}
