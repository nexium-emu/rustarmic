use rustarmic::{CpuContext, CpuFeatures, Error, ExitReason, Jit, JitConfig, Memory};

const CODE_BASE: u64 = 0x1000;
const BRK_0: u32 = 0xD420_0000;
const NOP: u32 = 0xD503_201F;

struct CodeMem {
    bytes: Vec<u8>,
}

impl Memory for CodeMem {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let off = addr.checked_sub(CODE_BASE)? as usize;
        let word = self.bytes.get(off..off.checked_add(4)?)?;
        Some(u32::from_le_bytes(word.try_into().ok()?))
    }
}

fn code(words: &[u32]) -> CodeMem {
    CodeMem {
        bytes: words.iter().flat_map(|word| word.to_le_bytes()).collect(),
    }
}

#[test]
fn bounded_run_does_not_execute_past_budget() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let mut mem = code(&[NOP, NOP, BRK_0]);
    let mut jit = Jit::new(JitConfig::default()).unwrap();

    let outcome = jit.run_bounded(&mut ctx, &mut mem, 1).unwrap();
    assert_eq!(outcome.reason, ExitReason::BudgetExhausted);
    assert_eq!(outcome.retired, 1);
    assert_eq!(ctx.pc, CODE_BASE + 4);

    let outcome = jit.run_bounded(&mut ctx, &mut mem, 1).unwrap();
    assert_eq!(outcome.reason, ExitReason::BudgetExhausted);
    assert_eq!(outcome.retired, 1);
    assert_eq!(ctx.pc, CODE_BASE + 8);

    let outcome = jit.run_bounded(&mut ctx, &mut mem, 1).unwrap();
    assert_eq!(outcome.reason, ExitReason::Brk(0));
    assert_eq!(outcome.retired, 1);
}

#[test]
fn halt_request_is_consumed_before_guest_execution() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.should_halt = 1;
    let mut mem = code(&[NOP, BRK_0]);
    let mut jit = Jit::new(JitConfig::default()).unwrap();

    let outcome = jit.run_bounded(&mut ctx, &mut mem, 10).unwrap();
    assert_eq!(outcome.reason, ExitReason::Stopped);
    assert_eq!(outcome.retired, 0);
    assert_eq!(ctx.pc, CODE_BASE);
    assert_eq!(ctx.should_halt, 0);
}

#[test]
fn rejects_hosts_below_sse41_baseline() {
    let result = Jit::new(JitConfig {
        host_features: Some(CpuFeatures::default()),
        ..JitConfig::default()
    });
    let error = match result {
        Ok(_) => panic!("SSE4.1-masked host must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::UnsupportedHost));
}
