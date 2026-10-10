use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE_BASE: u64 = 0x1000;

struct CodeMem([u32; 2]);
impl Memory for CodeMem {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let offset = addr.checked_sub(CODE_BASE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get((offset / 4) as usize).copied()
    }
}

fn opcode(castagnoli: bool, width: u32, rd: u32, rn: u32, rm: u32) -> u32 {
    0x1ac0_4000
        | ((width == 3) as u32) << 31
        | (castagnoli as u32) << 12
        | width << 10
        | rm << 16
        | rn << 5
        | rd
}

// Independent MSB-first polynomial division; the JIT uses reflected shifts.
fn reference(seed: u32, value: u64, width: u32, castagnoli: bool) -> u32 {
    let polynomial = if castagnoli { 0x1edc_6f41 } else { 0x04c1_1db7 };
    let mut crc = seed.reverse_bits();
    for bit in 0..(8 << width) {
        let feedback = (crc >> 31) ^ ((value >> bit) as u32 & 1);
        crc <<= 1;
        if feedback != 0 {
            crc ^= polynomial;
        }
    }
    crc.reverse_bits()
}

struct Harness {
    jit: Jit,
    memory: CodeMem,
    rd: usize,
    rn: usize,
    rm: usize,
    width: u32,
    castagnoli: bool,
}
impl Harness {
    fn new(castagnoli: bool, width: u32, rd: usize, rn: usize, rm: usize) -> Self {
        Self {
            jit: Jit::new(JitConfig::default()).unwrap(),
            memory: CodeMem([
                opcode(castagnoli, width, rd as u32, rn as u32, rm as u32),
                0xd420_0000,
            ]),
            rd,
            rn,
            rm,
            width,
            castagnoli,
        }
    }
    fn check(&mut self, seed: u32, value: u64) -> u32 {
        let mut context = CpuContext::default();
        context.pc = CODE_BASE;
        context.nzcv = 0xb;
        context.x.fill(0xfeed_face_1234_5678);
        if self.rn != 31 {
            context.x[self.rn] = 0xcafe_beef_0000_0000 | seed as u64;
        }
        if self.rm != 31 {
            context.x[self.rm] = value;
        }
        context
            .v
            .fill([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]);
        let original_x = context.x;
        let original_v = context.v;
        let actual_seed = if self.rn == 31 {
            0
        } else {
            context.x[self.rn] as u32
        };
        let actual_value = if self.rm == 31 { 0 } else { context.x[self.rm] };
        let expected = reference(actual_seed, actual_value, self.width, self.castagnoli);
        assert_eq!(
            self.jit.run(&mut context, &mut self.memory).unwrap(),
            ExitReason::Brk(0)
        );
        let mut expected_x = original_x;
        if self.rd != 31 {
            expected_x[self.rd] = expected as u64;
        }
        assert_eq!(
            context.x, expected_x,
            "opcode={:#010x} seed={actual_seed:#010x} value={actual_value:#018x}",
            self.memory.0[0]
        );
        assert_eq!(context.v, original_v);
        assert_eq!(context.nzcv, 0xb);
        expected
    }
}

#[test]
fn crc32_known_check_values_and_little_endian_chunking() {
    for (castagnoli, check) in [(false, 0xcbf4_3926), (true, 0xe306_9283)] {
        for widths in [&[0; 9][..], &[3, 0][..], &[2, 1, 0, 1][..]] {
            let mut accumulator = u32::MAX;
            let mut offset = 0usize;
            for &width in widths {
                let bytes = 1usize << width;
                let mut chunk = [0u8; 8];
                chunk[..bytes].copy_from_slice(&b"123456789"[offset..offset + bytes]);
                accumulator = Harness::new(castagnoli, width, 0, 1, 2)
                    .check(accumulator, u64::from_le_bytes(chunk));
                offset += bytes;
            }
            assert_eq!(offset, 9);
            assert_eq!(!accumulator, check);
        }
    }
}

#[test]
fn crc32_all_widths_random_and_edge_inputs() {
    let mut random = 0x4d59_5df4_d0f3_3173u64;
    let mut next = || {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        random
    };
    for castagnoli in [false, true] {
        for width in 0..4 {
            let mut harness = Harness::new(castagnoli, width, 0, 1, 2);
            for seed in [0, u32::MAX, 0x8000_0000, 0x5555_5555, 0xaaaa_aaaa] {
                for value in [0, u64::MAX, 0x8000_0000_0000_0000, 0x0123_4567_89ab_cdef] {
                    harness.check(seed, value);
                }
            }
            for _ in 0..128 {
                let seed = next() as u32;
                let value = next();
                harness.check(seed, value);
            }
        }
    }
}

#[test]
fn crc32_aliases_zero_register_and_w_destination() {
    for castagnoli in [false, true] {
        for width in 0..4 {
            for (rd, rn, rm) in [
                (0, 1, 2),
                (1, 1, 2),
                (2, 1, 2),
                (0, 31, 2),
                (0, 1, 31),
                (31, 1, 2),
                (0, 31, 31),
            ] {
                Harness::new(castagnoli, width, rd, rn, rm)
                    .check(0x89ab_cdef, 0xfedc_ba98_7654_3210);
            }
        }
    }
}

#[test]
fn wonder_crc32x_w10_w10_x11() {
    assert_eq!(opcode(false, 3, 10, 10, 11), 0x9acb_4d4a);
    Harness::new(false, 3, 10, 10, 11).check(0x1234_5678, 0x0123_4567_89ab_cdef);
}
