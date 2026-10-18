//! Instruction coverage: encodings a Cortex-A57 (Nintendo Switch)
//! executes must translate and compile instead of stopping the guest.
//!
//! Random encodings are decoded with disarm64 and filtered by feature set
//! (base ARMv8.0, FP, AdvSIMD, CRC32, AES, SHA-1/SHA-256, plus LSE which
//! Rustarmic also implements).  Each is translated, optimised and
//! compiled; when that fails, Unicorn's Cortex-A57 model decides whether
//! the encoding is allocated (disarm64 also decodes many reserved
//! encodings, which Rustarmic rightly rejects).
#![cfg(feature = "unicorn")]

use disarm64::decoder;
use disarm64_defn::InsnFeatureSet;
use disarm64_defn::defn::InsnOpcode;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustarmic::Error;
use rustarmic::backend::emit::emit_block;
use rustarmic::frontend::{TranslateOptions, translate_block_into};
use rustarmic::ir::Block;
use rustarmic::optimizer::optimize;
use std::collections::BTreeMap;
use unicorn_engine::unicorn_const::uc_error;
use unicorn_engine::{Arch, Mode, Prot, RegisterARM64, Unicorn};

fn on_switch(f: InsnFeatureSet) -> bool {
    matches!(
        f,
        InsnFeatureSet::V8
            | InsnFeatureSet::FP
            | InsnFeatureSet::SIMD
            | InsnFeatureSet::CRC
            | InsnFeatureSet::AES
            | InsnFeatureSet::SHA2
            | InsnFeatureSet::LSE
    )
}

/// Translates, optimises and compiles one instruction (followed by BRK).
fn compile(insn: u32) -> Result<(), Error> {
    let mut b = Block::new(0x1000);
    let opts = TranslateOptions {
        max_insts: 2,
        multiblock: false,
    };
    let mut fetch = |pc: u64| Some(if pc == 0x1000 { insn } else { 0xD420_0000 });
    translate_block_into(&mut b, 0x1000, &mut fetch, opts)?;
    optimize(&mut b);
    emit_block(&b).map(|_| ())
}

/// System and exception-generating instructions: the user-mode model
/// handles the ones a game can execute (SVC, BRK, MRS / MSR of EL0
/// registers, barriers, hints, cache maintenance) individually.
fn system(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
        "svc" | "hvc" | "smc" | "brk" | "hlt" | "eret" | "drps" | "dcps1" | "dcps2" | "dcps3" | "msr"
            | "mrs" | "sys" | "sysl" | "at" | "tlbi" | "dc" | "ic" | "wfe" | "wfi" | "sev" | "sevl"
            | "yield" | "hint" | "clrex" | "dsb" | "dmb" | "isb" | "nop" | "udf"
    )
}

struct Oracle {
    emu: Unicorn<'static, ()>,
}

impl Oracle {
    fn new() -> Self {
        let mut emu = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
        emu.ctl_set_cpu_model(0).unwrap(); // UC_CPU_ARM64_A57
        emu.mem_map(0x1000, 0x1000, Prot::ALL).unwrap();
        emu.mem_map(0x10_0000, 0x20_0000, Prot::ALL).unwrap();
        let cpacr = emu.reg_read(RegisterARM64::CPACR_EL1).unwrap_or(0);
        let _ = emu.reg_write(RegisterARM64::CPACR_EL1, cpacr | (0b11 << 20));
        Self { emu }
    }

    /// Does the A57 model execute `insn` (rather than raise UNDEFINED)?
    fn allocated(&mut self, insn: u32) -> bool {
        use RegisterARM64::*;
        let emu = &mut self.emu;
        emu.mem_write(0x1000, &insn.to_le_bytes()).unwrap();
        emu.ctl_flush_tb().unwrap();
        for r in [
            X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18,
            X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, SP,
        ] {
            emu.reg_write(r, 0x18_0000).unwrap();
        }
        !matches!(emu.emu_start(0x1000, 0x1004, 0, 1), Err(uc_error::EXCEPTION))
    }
}

fn scan(words: u64, seed: u64) -> BTreeMap<String, (u64, u32, String)> {
    std::panic::set_hook(Box::new(|_| {}));
    let mut oracle = Oracle::new();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut gaps: BTreeMap<String, (u64, u32, String)> = BTreeMap::new();
    for _ in 0..words {
        let w: u32 = rng.random();
        let Some(op) = decoder::decode(w) else {
            continue;
        };
        let def = op.definition();
        if !on_switch(def.feature_set) || system(def.mnemonic) {
            continue;
        }
        let err = match std::panic::catch_unwind(|| compile(w)) {
            Ok(Ok(())) => continue,
            Ok(Err(e)) => format!("{e:?}"),
            Err(p) => format!(
                "PANIC {}",
                p.downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default()
            ),
        };
        if !err.starts_with("PANIC") && !oracle.allocated(w) {
            continue;
        }
        let key = format!("{} ({:?})", def.mnemonic, def.class);
        let e = gaps.entry(key).or_insert((0, w, err));
        e.0 += 1;
    }
    let _ = std::panic::take_hook();
    gaps
}

/// Investigation helper: lists every remaining gap with an example.
/// `RUSTARMIC_SCAN_WORDS=<n>` sets the sample size.
#[test]
#[ignore = "investigation helper"]
fn list_coverage_gaps() {
    let n = std::env::var("RUSTARMIC_SCAN_WORDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5_000_000);
    for (k, (count, w, e)) in scan(n, 0x5CA1) {
        eprintln!("{k}: {count} e.g. {w:08x} -> {e}");
    }
}

/// No allocated ARMv8.0 / FP / AdvSIMD / CRC32 / AES / SHA-1 / SHA-256
/// encoding stops the guest as unsupported, and no translator panics.
#[test]
fn switch_instructions_translate() {
    let n = std::env::var("RUSTARMIC_SCAN_WORDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let gaps = scan(n, 0xC0DE_5CA1);
    let list: Vec<String> = gaps
        .iter()
        .map(|(k, (count, w, e))| format!("{k}: {count} e.g. {w:08x} -> {e}"))
        .collect();
    assert!(list.is_empty(), "unsupported instructions:\n{}", list.join("\n"));
}
