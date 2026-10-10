use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const CANARY: u64 = 0xface_1234_5678_9abc;

struct CodeMem([u32; 2]);
impl Memory for CodeMem {
    fn fetch_inst(&mut self, address: u64) -> Option<u32> {
        let offset = address.checked_sub(CODE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get((offset / 4) as usize).copied()
    }
}

fn bytes(vector: [u64; 2]) -> [u8; 16] {
    let mut output = [0; 16];
    output[..8].copy_from_slice(&vector[0].to_le_bytes());
    output[8..].copy_from_slice(&vector[1].to_le_bytes());
    output
}
fn vector(data: [u8; 16]) -> [u64; 2] {
    [
        u64::from_le_bytes(data[..8].try_into().unwrap()),
        u64::from_le_bytes(data[8..].try_into().unwrap()),
    ]
}

fn oracle(data: [u64; 2], shifts: [u64; 2], size: u32, q: bool, signed: bool) -> [u64; 2] {
    let (data, shifts) = (bytes(data), bytes(shifts));
    let element_bytes = 1usize << size;
    let width = element_bytes * 8;
    let mask = (1u128 << width) - 1;
    let mut result = [0; 16];
    for lane in 0..(if q { 16 } else { 8 }) / element_bytes {
        let offset = lane * element_bytes;
        let mut raw = [0; 8];
        raw[..element_bytes].copy_from_slice(&data[offset..offset + element_bytes]);
        let value = u64::from_le_bytes(raw) as u128;
        // Only one byte is read, even if the other count bytes are nonzero.
        let count = shifts[offset] as i8 as i32;
        let output = if count >= 0 {
            if count as usize >= width {
                0
            } else {
                (value << count) & mask
            }
        } else if signed {
            let value = if value & (1u128 << (width - 1)) != 0 {
                value as i128 - (1i128 << width)
            } else {
                value as i128
            };
            let shifted = if (-count) as usize >= width {
                if value < 0 {
                    -1
                } else {
                    0
                }
            } else {
                value >> -count
            };
            shifted as u128 & mask
        } else if (-count) as usize >= width {
            0
        } else {
            value >> -count
        };
        result[offset..offset + element_bytes]
            .copy_from_slice(&(output as u64).to_le_bytes()[..element_bytes]);
    }
    vector(result)
}

fn opcode(size: u32, q: bool, signed: bool, rn: usize, rm: usize, rd: usize) -> u32 {
    0x0e20_4400
        | (u32::from(!signed) << 29)
        | (u32::from(q) << 30)
        | (size << 22)
        | ((rm as u32) << 16)
        | ((rn as u32) << 5)
        | rd as u32
}

fn cases(size: u32, count: u8, pattern: usize) -> ([u64; 2], [u64; 2]) {
    let element_bytes = 1usize << size;
    let width = element_bytes * 8;
    let mask = u64::MAX >> (64 - width);
    let sign = 1u64 << (width - 1);
    let mut values = [0; 16];
    let mut shifts = [0; 16];
    for lane in 0..16 / element_bytes {
        let value = match pattern {
            0 => 0,
            1 => 1,
            2 => mask,
            3 => sign,
            4 => sign - 1,
            _ => {
                0x96a5_3c7e_d18b_42f0u64.rotate_left((lane * 13 + count as usize % 64) as u32)
                    & mask
            }
        };
        let shift = (0x5a96_d183_7cb2_e400u64.rotate_left((lane * 7) as u32) & !0xff)
            | count.wrapping_add((lane * 31) as u8) as u64;
        let offset = lane * element_bytes;
        values[offset..offset + element_bytes]
            .copy_from_slice(&value.to_le_bytes()[..element_bytes]);
        shifts[offset..offset + element_bytes]
            .copy_from_slice(&shift.to_le_bytes()[..element_bytes]);
    }
    (vector(values), vector(shifts))
}

fn check(size: u32, q: bool, signed: bool, rn: usize, rm: usize, rd: usize) {
    let word = opcode(size, q, signed, rn, rm, rd);
    let mut memory = CodeMem([word, 0xd420_0000]);
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    for count in 0..=255u8 {
        for pattern in 0..6 {
            let (data, shifts) = cases(size, count, pattern);
            let mut context = CpuContext::default();
            context.pc = CODE;
            context.x.fill(CANARY);
            context.sp = CANARY;
            context.nzcv = 0xa;
            context.v.fill([CANARY, CANARY]);
            context.v[rn] = data;
            context.v[rm] = shifts;
            let mut expected = context.v;
            expected[rd] = oracle(context.v[rn], context.v[rm], size, q, signed);
            assert_eq!(
                jit.run(&mut context, &mut memory).unwrap(),
                ExitReason::Brk(0)
            );
            assert_eq!(
                context.v, expected,
                "word={word:#010x} count={count:#04x} pattern={pattern}"
            );
            assert!(context.x.iter().all(|&value| value == CANARY));
            assert_eq!(context.sp, CANARY);
            assert_eq!(context.nzcv, 0xa);
        }
    }
}

fn family(signed: bool) {
    for size in 0..4 {
        for q in [false, true] {
            if size == 3 && !q {
                continue;
            }
            for (rn, rm, rd) in [
                (1, 2, 0),
                (1, 2, 1),
                (1, 2, 2),
                (1, 1, 0),
                (1, 1, 1),
                (31, 2, 0),
                (1, 31, 31),
            ] {
                check(size, q, signed, rn, rm, rd);
            }
        }
    }
}

#[test]
fn ushl_all_legal_vector_formats_and_signed_low_byte_counts() {
    family(false);
}

#[test]
fn sshl_all_legal_vector_formats_and_signed_low_byte_counts() {
    family(true);
}

#[test]
fn exact_wonder_ushl_v2_2s_v2_v3() {
    assert_eq!(opcode(2, false, false, 2, 3, 2), 0x2ea3_4442);
    check(2, false, false, 2, 3, 2);
}

#[test]
fn reserved_1d_vector_shift_is_rejected() {
    for signed in [false, true] {
        let mut memory = CodeMem([opcode(3, false, signed, 1, 2, 0), 0xd420_0000]);
        let mut context = CpuContext::default();
        context.pc = CODE;
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        let outcome = jit.run_bounded(&mut context, &mut memory, 2).unwrap();
        assert!(matches!(outcome.stop, StopReason::Unsupported(_)));
        assert_eq!(outcome.retired, 0);
    }
}

#[test]
fn oracle_manual_boundaries_and_ignored_high_count_bytes() {
    let input = [0x8000_0001_ffff_ffff, 0xdead_beef_1234_5678];
    assert_eq!(
        oracle(input, [0x1234_5600_ffff_ff01, 0], 2, false, false),
        [0x8000_0001_ffff_fffe, 0]
    );
    assert_eq!(
        oracle(input, [0x7654_3280_abcdefe0, 0], 2, false, false),
        [0, 0]
    );
    assert_eq!(
        oracle(input, [0x7654_3280_abcdefe0, 0], 2, false, true),
        [u64::MAX, 0]
    );
    assert_eq!(
        oracle(
            [0x8000_0000_0000_0001, 0],
            [0x1234_5678_9abc_debf, 0],
            3,
            true,
            true
        ),
        [u64::MAX, 0]
    );
    assert_eq!(
        oracle([1, 0], [0x1234_5678_9abc_de40, 0], 3, true, false),
        [0, 0]
    );
}
