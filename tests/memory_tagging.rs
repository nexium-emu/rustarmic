use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE_BASE: u64 = 0x1000;
const DATA_BASE: u64 = 0x20_0000;
const BASE: usize = 0x100;
const BRK_0: u32 = 0xD420_0000;
const TAGGED: u64 = 0x0A00_0000_0000_0000;
const POST: u32 = 0b01;
const OFFSET: u32 = 0b10;
const PRE: u32 = 0b11;

struct CodeMem(Vec<u8>);

impl Memory for CodeMem {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let off = addr.checked_sub(CODE_BASE)? as usize;
        let bytes = self.0.get(off..off.checked_add(4)?)?;
        Some(u32::from_le_bytes(bytes.try_into().ok()?))
    }
}

fn run(words: &[u32], data: &[(usize, u64)], setup: impl FnOnce(&mut CpuContext)) -> (CpuContext, Vec<u8>) {
    let code = words.iter().chain([BRK_0].iter()).flat_map(|word| word.to_le_bytes()).collect();
    let mut backing = vec![0xAAu8; 0x1000];
    for (offset, value) in data {
        backing[*offset..*offset + 8].copy_from_slice(&value.to_le_bytes());
    }
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE + BASE as u64;
    ctx.mem_base = backing.as_mut_ptr();
    ctx.mem_base_va = DATA_BASE;
    ctx.mem_size = backing.len() as u64;
    setup(&mut ctx);
    let mut jit = Jit::new(JitConfig {
        use_fastmem: true,
        ..JitConfig::default()
    })
    .unwrap();
    assert_eq!(jit.run(&mut ctx, &mut CodeMem(code)).unwrap(), ExitReason::Brk(0));
    (ctx, backing)
}

fn tag_store(opc: u32, granules: i32, mode: u32, rn: u32, rt: u32) -> u32 {
    0xD920_0000 | (opc << 22) | (((granules as u32) & 0x1FF) << 12) | (mode << 10) | (rn << 5) | rt
}

fn stgp(mode: u32, granules: i32, rt2: u32, rn: u32, rt: u32) -> u32 {
    0x6800_0000 | (mode << 23) | (((granules as u32) & 0x7F) << 15) | (rt2 << 10) | (rn << 5) | rt
}

fn ldra(key_b: bool, doublewords: i32, writeback: bool, rn: u32, rt: u32) -> u32 {
    let imm = (doublewords as u32) & 0x3FF;
    0xF820_0400
        | (u32::from(key_b) << 23)
        | ((imm >> 9) << 22)
        | ((imm & 0x1FF) << 12)
        | (u32::from(writeback) << 11)
        | (rn << 5)
        | rt
}

fn word(memory: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(memory[offset..offset + 8].try_into().unwrap())
}

#[test]
fn tag_stores_leave_data_alone_and_write_back_the_address() {
    assert_eq!(tag_store(0, 0, OFFSET, 8, 8), 0xD920_0908);
    for opc in [0, 2] {
        for (mode, moved) in [(OFFSET, 0), (POST, 32), (PRE, 32)] {
            let (ctx, memory) = run(&[tag_store(opc, 2, mode, 0, 1)], &[], |ctx| ctx.x[1] = TAGGED);
            assert_eq!(ctx.x[0], DATA_BASE + BASE as u64 + moved, "opc={opc} mode={mode}");
            assert_eq!(ctx.x[1], TAGGED);
            assert!(memory.iter().all(|byte| *byte == 0xAA), "opc={opc} mode={mode}");
        }
    }
}

#[test]
fn zeroing_tag_stores_clear_one_or_two_granules() {
    for (opc, bytes) in [(1u32, 16usize), (3, 32)] {
        let (ctx, memory) = run(&[tag_store(opc, -1, PRE, 0, 31)], &[], |_| {});
        let start = BASE - 16;
        assert_eq!(ctx.x[0], DATA_BASE + start as u64);
        assert!(memory[start..start + bytes].iter().all(|byte| *byte == 0));
        assert!(memory[..start].iter().chain(&memory[start + bytes..]).all(|byte| *byte == 0xAA));
    }
    let (ctx, memory) = run(&[tag_store(1, 1, POST, 0, 31)], &[], |_| {});
    assert_eq!(ctx.x[0], DATA_BASE + BASE as u64 + 16);
    assert!(memory[BASE..BASE + 16].iter().all(|byte| *byte == 0));
    assert!(memory[BASE + 16..].iter().all(|byte| *byte == 0xAA));
}

#[test]
fn ldg_inserts_tag_zero() {
    let (ctx, _) = run(&[0xD960_0001], &[], |ctx| ctx.x[1] = TAGGED | 0x1234);
    assert_eq!(ctx.x[1], 0x1234);
}

#[test]
fn stgp_stores_the_pair_and_writes_back() {
    for (mode, data_at, moved_to) in [(OFFSET, BASE + 16, BASE), (POST, BASE, BASE + 16), (PRE, BASE + 16, BASE + 16)] {
        let (ctx, memory) = run(&[stgp(mode, 1, 2, 0, 1)], &[], |ctx| {
            ctx.x[1] = 0x1111_2222_3333_4444;
            ctx.x[2] = 0x5555_6666_7777_8888;
        });
        assert_eq!(ctx.x[0], DATA_BASE + moved_to as u64, "mode={mode}");
        assert_eq!(word(&memory, data_at), 0x1111_2222_3333_4444, "mode={mode}");
        assert_eq!(word(&memory, data_at + 8), 0x5555_6666_7777_8888, "mode={mode}");
    }
    let (_, memory) = run(&[stgp(OFFSET, 0, 31, 0, 31)], &[], |_| {});
    assert_eq!((word(&memory, BASE), word(&memory, BASE + 8)), (0, 0));
}

#[test]
fn tag_arithmetic_produces_tag_zero() {
    let base = TAGGED | 0x1000;
    let (ctx, _) = run(&[0x9AC0_1001 | (2 << 16)], &[], |ctx| ctx.x[0] = base);
    assert_eq!(ctx.x[1], 0x1000);
    let (ctx, _) = run(&[0x9AC0_1403 | (2 << 16)], &[], |ctx| {
        ctx.x[0] = base;
        ctx.x[2] = 0x8000;
    });
    assert_eq!(ctx.x[3], 0x8000 | (1 << 0xA));
    let (ctx, _) = run(&[0x9180_0C01 | (3 << 16), 0xD180_0402 | (1 << 16)], &[], |ctx| ctx.x[0] = base);
    assert_eq!((ctx.x[1], ctx.x[2]), (0x1030, 0xFF0));
}

#[test]
fn subp_ignores_tags_and_subps_sets_flags() {
    let (ctx, _) = run(&[0x9AC0_0002 | (1 << 16), 0xBAC0_0023], &[], |ctx| {
        ctx.x[0] = TAGGED | 0x3000;
        ctx.x[1] = 0x0F00_0000_0000_1000;
    });
    assert_eq!(ctx.x[2], 0x2000);
    assert_eq!(ctx.x[3], (-0x2000i64) as u64);
    assert_eq!(ctx.nzcv, 0x8);
}

#[test]
fn ldraa_and_ldrab_load_with_scaled_offsets() {
    let data = [(BASE + 0x18, 0x1234_5678_9ABC_DEF0), (BASE - 0x20, 0x0FED_CBA9_8765_4321)];
    let (ctx, _) = run(&[ldra(false, 3, false, 0, 1), ldra(true, -4, true, 0, 2)], &data, |_| {});
    assert_eq!(ctx.x[1], 0x1234_5678_9ABC_DEF0);
    assert_eq!(ctx.x[2], 0x0FED_CBA9_8765_4321);
    assert_eq!(ctx.x[0], DATA_BASE + BASE as u64 - 0x20);
}
