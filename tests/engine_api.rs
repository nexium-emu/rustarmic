use rustarmic::{Engine, EngineConfig, FlatMemory, StopReason};

const PC: u64 = 0x1000;

#[test]
fn engine_defaults_are_bounded_and_accurate() {
    let config = EngineConfig::default();
    assert_eq!(config.max_block_insts, 64);
    assert_eq!(config.code_cache_bytes, 256 * 1024 * 1024);
    assert_eq!(config.fp_mode, rustarmic::FpMode::Accurate);
}

#[test]
fn engine_zero_budget_executes_nothing() {
    let engine = Engine::new(EngineConfig::default()).unwrap();
    let mut memory = FlatMemory::new(PC, 0x1000);
    memory.write_u32(PC, 0xD4200000); // BRK #0
    let mut ctx = rustarmic::CpuContext::default();
    ctx.pc = PC;
    let outcome = engine.run(&mut ctx, &mut memory, 0).unwrap();
    assert_eq!(outcome.retired, 0);
    assert_eq!(outcome.stop, StopReason::BudgetExhausted);
    assert_eq!(ctx.pc, PC);
}

#[test]
fn engine_step_reports_one_retirement() {
    let engine = Engine::new(EngineConfig::default()).unwrap();
    let mut memory = FlatMemory::new(PC, 0x1000);
    memory.write_u32(PC, 0xD503201F); // NOP
    memory.write_u32(PC + 4, 0xD4200000); // BRK #0
    let mut ctx = rustarmic::CpuContext::default();
    ctx.pc = PC;
    let outcome = engine.step(&mut ctx, &mut memory).unwrap();
    assert_eq!(outcome.retired, 1);
    assert_eq!(outcome.stop, StopReason::BudgetExhausted);
    assert_eq!(ctx.pc, PC + 4);
}

#[test]
fn unsupported_decode_is_a_structured_stop() {
    let engine = Engine::new(EngineConfig::default()).unwrap();
    let mut memory = FlatMemory::new(PC, 0x1000);
    memory.write_u32(PC, 0); // reserved/unsupported A64 encoding
    let mut ctx = rustarmic::CpuContext::default();
    ctx.pc = PC;
    let outcome = engine.run(&mut ctx, &mut memory, 1).unwrap();
    assert!(matches!(outcome.stop, StopReason::Unsupported(_)));
    assert_eq!(outcome.retired, 0);
}

#[test]
fn execute_fetch_miss_is_a_precise_memory_fault() {
    let engine = Engine::new(EngineConfig::default()).unwrap();
    let mut memory = FlatMemory::new(PC, 4);
    let mut ctx = rustarmic::CpuContext::default();
    ctx.pc = PC + 4;
    let outcome = engine.run(&mut ctx, &mut memory, 1).unwrap();
    match outcome.stop {
        StopReason::MemoryFault(fault) => {
            assert_eq!(fault.pc, PC + 4);
            assert_eq!(fault.address, PC + 4);
            assert_eq!(fault.size, 4);
        }
        other => panic!("expected memory fault, got {other:?}"),
    }
}
