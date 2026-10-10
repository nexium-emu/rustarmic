use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const BRK: u32 = 0xd420_0000;
struct Code(Vec<u32>);
impl Memory for Code {
    fn fetch_inst(&mut self, address: u64) -> Option<u32> {
        let offset = address.checked_sub(CODE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get(offset as usize / 4).copied()
    }
}

fn opcode(size: u32, lane: u32, rn: u32, rd: u32) -> u32 {
    let imm5 = (lane << (size + 1)) | (1 << size);
    0x5e00_0400 | (imm5 << 16) | (rn << 5) | rd
}

fn vector(bytes: [u8; 16]) -> [u64; 2] {
    [
        u64::from_le_bytes(bytes[..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..].try_into().unwrap()),
    ]
}

fn bytes(vector: [u64; 2]) -> [u8; 16] {
    let mut result = [0; 16];
    result[..8].copy_from_slice(&vector[0].to_le_bytes());
    result[8..].copy_from_slice(&vector[1].to_le_bytes());
    result
}

fn expected(source: [u64; 2], size: u32, lane: u32) -> [u64; 2] {
    // An independent byte-slice oracle, without decoder/extraction helpers.
    let width = 1usize << size;
    let source = bytes(source);
    let mut result = [0; 16];
    result[..width].copy_from_slice(&source[lane as usize * width..(lane as usize + 1) * width]);
    vector(result)
}

fn context() -> CpuContext {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.x.fill(GUARD);
    ctx.sp = GUARD;
    ctx.nzcv = 0xb;
    ctx.fpcr = 0x0340_0000;
    ctx.fpsr = 0x0800_009f;
    for (reg, value) in ctx.v.iter_mut().enumerate() {
        *value = vector(std::array::from_fn(|byte| {
            (reg * 37 + byte * 13 + 0x91) as u8
        }));
    }
    ctx
}

fn assert_guards(ctx: &CpuContext) {
    assert_eq!(ctx.x, [GUARD; 31]);
    assert_eq!(ctx.sp, GUARD);
    assert_eq!(ctx.nzcv, 0xb);
    assert_eq!(ctx.fpcr, 0x0340_0000);
    assert_eq!(ctx.fpsr, 0x0800_009f);
}

#[test]
fn every_width_lane_alias_and_v31_clears_all_unused_destination_bits() {
    let patterns = [
        [0; 16],
        [0xff; 16],
        std::array::from_fn(|byte| (byte * 17 + 0x80) as u8),
        std::array::from_fn(|byte| if byte & 1 == 0 { 0x55 } else { 0xaa }),
        std::array::from_fn(|byte| ((byte * 101 + 7) ^ (byte * 13 + 0xb3)) as u8),
    ];
    let mut cases = 0;
    for size in 0..4 {
        for lane in 0..(16 >> size) {
            for (rn, rd) in [(2, 1), (2, 2), (31, 31), (31, 0), (0, 31)] {
                let instruction = opcode(size, lane, rn, rd);
                let mut memory = Code(vec![instruction, BRK]);
                let mut jit = Jit::new(JitConfig::default()).unwrap();
                for source in patterns {
                    let mut ctx = context();
                    ctx.v[rn as usize] = vector(source);
                    let mut wanted = ctx.v;
                    wanted[rd as usize] = expected(ctx.v[rn as usize], size, lane);
                    let result = jit.run_bounded(&mut ctx, &mut memory, 2).unwrap();
                    assert_eq!(result.stop, StopReason::Brk(0), "{instruction:#x}");
                    assert_eq!(result.retired, 2);
                    assert_eq!(ctx.v, wanted, "{instruction:#x} source={source:?}");
                    assert_guards(&ctx);
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 750);
}

#[test]
fn exact_wonder_scalar_s_lane_two_and_self_alias() {
    assert_eq!(opcode(2, 2, 2, 1), 0x5e14_0441);
    for rd in [1, 2] {
        let mut ctx = context();
        ctx.v[2] = [0x1111_2222_3333_4444, 0x5555_6666_89ab_cdef];
        let mut wanted = ctx.v;
        wanted[rd as usize] = [0x89ab_cdef, 0];
        let mut memory = Code(vec![opcode(2, 2, 2, rd), BRK]);
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        let result = jit.run_bounded(&mut ctx, &mut memory, 2).unwrap();
        assert_eq!(result.stop, StopReason::Brk(0));
        assert_eq!(ctx.v, wanted);
        assert_guards(&ctx);
    }
}

#[test]
fn zero_and_reserved_128bit_imm5_reject_without_retiring_or_mutation() {
    for imm5 in [0, 16] {
        let instruction = 0x5e00_0400 | (imm5 << 16) | (31 << 5) | 31;
        let mut memory = Code(vec![instruction]);
        let mut ctx = context();
        let original_v = ctx.v;
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        let result = jit.run_bounded(&mut ctx, &mut memory, 1).unwrap();
        assert!(
            matches!(result.stop, StopReason::Unsupported(info) if info.opcode == instruction && info.pc == CODE)
        );
        assert_eq!(result.retired, 0);
        assert_eq!(ctx.pc, CODE);
        assert_eq!(ctx.v, original_v);
        assert_guards(&ctx);
    }
}

#[test]
fn mixed_lane_copy_chain_observes_prior_writes_and_source_upper_half() {
    let sequence = [
        (3, 1, 31, 0),
        (2, 1, 0, 31),
        (1, 0, 31, 31),
        (0, 7, 31, 2),
        (0, 15, 1, 3),
        (1, 7, 1, 4),
        (2, 3, 1, 5),
        (3, 1, 1, 6),
    ];
    let mut instructions = sequence
        .iter()
        .map(|&(s, l, n, d)| opcode(s, l, n, d))
        .collect::<Vec<_>>();
    instructions.push(BRK);
    let mut memory = Code(instructions);
    let mut ctx = context();
    let mut wanted = ctx.v;
    for &(size, lane, rn, rd) in &sequence {
        wanted[rd as usize] = expected(wanted[rn as usize], size, lane);
    }
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    let result = jit.run_bounded(&mut ctx, &mut memory, 9).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0));
    assert_eq!(result.retired, 9);
    assert_eq!(ctx.v, wanted);
    assert_guards(&ctx);
}
