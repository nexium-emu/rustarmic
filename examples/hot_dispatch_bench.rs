//! CPU-only hot dispatch measurement. Run only on an idle host.
//! The first run prints fixed counts for an exactly matched comparison.
//! Defaults retain the original 1/2/32/128-block measurements. Larger capacity
//! experiments can select --working-sets 256,512,1024,2048,4096. Lookup-only
//! --lookup-stride accepts up to 64 MiB without allocating any guest memory.
//! For a capacity-aware fold (pc>>2)^(pc>>(2+index_bits)), forced-collision
//! strides are 0x40000/0x400000/0x4000000 for 256/1024/4096 entries; pass
//! their decimal values 262144/4194304/67108864 through --lookup-stride.

use rustarmic::jit::code_cache::CodeCache;
use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};
use std::hint::black_box;
use std::time::{Duration, Instant};

const CODE: u64 = 0x10_0000;
const SETS: [usize; 4] = [1, 2, 32, 128];
const TRIALS: usize = 5;
const GUARD: u64 = 0x1234_5678_9abc_def0;
const WARM_BLOCKS: u64 = 16_384;
const PROBE_COUNT: u64 = 1_048_576;
const MAX_WORKING_SET: usize = 4096;
const MAX_LOOKUP_STRIDE: u64 = 64 * 1024 * 1024;

struct Program {
    words: Vec<u32>,
    fetches: u64,
}

impl Memory for Program {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        self.fetches += 1;
        let offset = addr.checked_sub(CODE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.words.get((offset / 4) as usize).copied()
    }
}

fn ring(working_set: usize) -> Program {
    let mut words = Vec::new();
    for index in 0..working_set {
        let branch_pc = CODE + index as u64 * 16 + 12;
        let next_pc = CODE + ((index + 1) % working_set) as u64 * 16;
        let branch_delta = (next_pc as i64 - branch_pc as i64) / 4;
        words.extend([
            0x9100_0400, // ADD X0,X0,#1.
            0xca04_0042, // EOR X2,X2,X4.
            0x9100_1463, // ADD X3,X3,#5.
            0x1400_0000 | (branch_delta as u32 & 0x03ff_ffff),
        ]);
    }
    Program { words, fetches: 0 }
}

fn context() -> CpuContext {
    let mut ctx = CpuContext::default();
    for (index, value) in ctx.x.iter_mut().enumerate() {
        *value = GUARD ^ index as u64;
    }
    ctx.x[0] = 0;
    ctx.x[2] = !GUARD;
    ctx.x[3] = 7;
    ctx.x[4] = GUARD;
    ctx.v = [[GUARD, !GUARD]; 32];
    ctx.pc = CODE;
    ctx.sp = GUARD;
    ctx.nzcv = 0xf;
    ctx.fpcr = 0x0300_0000;
    ctx.fpsr = 0x0800_009f;
    ctx
}

fn assert_dispatch_state(ctx: &CpuContext, blocks: u64, working_set: usize) {
    let mut expected = context().x;
    expected[0] = blocks;
    expected[2] = !GUARD ^ if blocks & 1 != 0 { GUARD } else { 0 };
    expected[3] = 7 + blocks * 5;
    assert_eq!(ctx.x, expected);
    assert_eq!(ctx.v, [[GUARD, !GUARD]; 32]);
    assert_eq!(ctx.pc, CODE + (blocks % working_set as u64) * 16);
    assert_eq!(
        (ctx.sp, ctx.nzcv, ctx.fpcr, ctx.fpsr),
        (GUARD, 0xf, 0x0300_0000, 0x0800_009f)
    );
    assert_eq!((ctx.mem_fault, ctx.should_halt), (0, 0));
}

fn dispatch_trial(
    jit: &mut Jit,
    program: &mut Program,
    blocks: u64,
    working_set: usize,
) -> Duration {
    let mut ctx = context();
    let fetches = program.fetches;
    let started = Instant::now();
    let outcome = jit.run_bounded(&mut ctx, program, blocks * 4).unwrap();
    let elapsed = started.elapsed();
    assert_eq!(outcome.stop, StopReason::BudgetExhausted);
    assert_eq!(outcome.retired, blocks * 4);
    assert_eq!(
        program.fetches, fetches,
        "timed dispatch must not compile/fetch"
    );
    assert_dispatch_state(&ctx, blocks, working_set);
    elapsed
}

fn iteration_count(probe: Duration, target: Duration, working_set: usize) -> u64 {
    let target_ns = target.as_nanos();
    let probe_ns = probe.as_nanos().max(1);
    let raw = (u128::from(PROBE_COUNT) * target_ns).div_ceil(probe_ns);
    let count = raw.clamp(working_set as u128, 1 << 29) as u64;
    count.div_ceil(working_set as u64) * working_set as u64
}

fn report(kind: &str, working_set: usize, count: u64, trials: &[Duration]) {
    let mut millis: Vec<f64> = trials
        .iter()
        .map(|time| time.as_secs_f64() * 1000.0)
        .collect();
    let mean = millis.iter().sum::<f64>() / millis.len() as f64;
    millis.sort_by(f64::total_cmp);
    let shortest = millis[0];
    println!(
        "kind={kind} working_set={working_set} count={count} mean_ms={mean:.6} median_ms={:.6} min_ms={shortest:.6}",
        millis[millis.len() / 2]
    );
    if shortest < 100.0 {
        println!(
            "note={kind}/{working_set}: at least one window is below100ms; use a larger fixed count for a matched rerun"
        );
    }
}

fn dispatch_bench(counts: Option<&[u64]>, working_sets: &[usize], target: Duration) -> Vec<u64> {
    let mut selected = vec![0; working_sets.len()];
    for (slot, working_set) in working_sets.iter().copied().enumerate() {
        let mut program = ring(working_set);
        let before = program.words.clone();
        let mut config = JitConfig::default();
        // Keep the original default cache size; larger rings must stay fully
        // warm rather than measuring cache rollover/recompilation by accident.
        config.code_cache_bytes = (1024usize * 1024).max(working_set * 4096);
        config.translate.multiblock = false;
        let mut jit = Jit::new(config).unwrap();
        let mut warm = context();
        let warmed = jit
            .run_bounded(&mut warm, &mut program, WARM_BLOCKS * 4)
            .unwrap();
        assert_eq!(warmed.stop, StopReason::BudgetExhausted);
        assert_eq!(warmed.retired, WARM_BLOCKS * 4);
        assert_dispatch_state(&warm, WARM_BLOCKS, working_set);
        for index in 0..working_set {
            assert_eq!(
                jit.cache.lookup_meta(CODE + index as u64 * 16).unwrap().1,
                4
            );
        }
        let count = counts.map(|values| values[slot]).unwrap_or_else(|| {
            let probe = dispatch_trial(&mut jit, &mut program, PROBE_COUNT, working_set);
            iteration_count(probe, target, working_set)
        });
        selected[slot] = count;
        let mut trials = Vec::new();
        for trial in 0..TRIALS {
            let elapsed = dispatch_trial(&mut jit, &mut program, count, working_set);
            println!(
                "kind=dispatch working_set={working_set} trial={} blocks={count} retired={} elapsed_ms={:.6} ns_per_block={:.3}",
                trial + 1,
                count * 4,
                elapsed.as_secs_f64() * 1000.0,
                elapsed.as_secs_f64() * 1_000_000_000.0 / count as f64
            );
            trials.push(elapsed);
        }
        assert_eq!(program.words, before);
        report("dispatch", working_set, count, &trials);
    }
    selected
}

fn lookup_trial(cache: &CodeCache, entries: &[(u64, *const u8, u32)], count: u64) -> Duration {
    let started = Instant::now();
    for index in 0..count {
        let &(pc, expected_pointer, expected_count) =
            &entries[index as usize & (entries.len() - 1)];
        let result = black_box(cache).lookup_meta(black_box(pc)).unwrap();
        assert_eq!(result, (expected_pointer, expected_count));
        black_box(result);
    }
    started.elapsed()
}

fn lookup_bench(
    counts: Option<&[u64]>,
    working_sets: &[usize],
    target: Duration,
    stride: u64,
) -> Vec<u64> {
    let mut selected = vec![0; working_sets.len()];
    for (slot, working_set) in working_sets.iter().copied().enumerate() {
        let mut cache = CodeCache::new(1024 * 1024).unwrap();
        let mut entries = Vec::new();
        for index in 0..working_set {
            let pc = CODE + index as u64 * stride;
            let insns = index as u32 % 4 + 1;
            let pointer = cache
                .install(pc, pc + u64::from(insns) * 4, insns, &[0xc3], &[], 0, true)
                .unwrap();
            entries.push((pc, pointer, insns));
        }
        lookup_trial(&cache, &entries, PROBE_COUNT);
        let count = counts.map(|values| values[slot]).unwrap_or_else(|| {
            iteration_count(
                lookup_trial(&cache, &entries, PROBE_COUNT),
                target,
                working_set,
            )
        });
        selected[slot] = count;
        let mut trials = Vec::new();
        for trial in 0..TRIALS {
            let elapsed = lookup_trial(&cache, &entries, count);
            println!(
                "kind=lookup working_set={working_set} trial={} lookups={count} elapsed_ms={:.6} ns_per_lookup={:.3}",
                trial + 1,
                elapsed.as_secs_f64() * 1000.0,
                elapsed.as_secs_f64() * 1_000_000_000.0 / count as f64
            );
            trials.push(elapsed);
        }
        report("lookup", working_set, count, &trials);
    }
    selected
}

fn parse_working_sets(text: &str) -> Result<Vec<usize>, String> {
    let values: Vec<usize> = text
        .split(',')
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| "invalid working set".to_string())
        })
        .collect::<Result<_, _>>()?;
    if values.is_empty()
        || values
            .iter()
            .any(|&value| value > MAX_WORKING_SET || !value.is_power_of_two())
        || values.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("working sets must be increasing powers of two in 1..=4096".into());
    }
    Ok(values)
}

fn parse_counts_for_sets(text: &str, working_sets: &[usize]) -> Result<Vec<u64>, String> {
    let values: Vec<u64> = text
        .split(',')
        .map(|value| {
            value
                .parse::<u64>()
                .map_err(|_| "invalid count".to_string())
        })
        .collect::<Result<_, _>>()?;
    if values.len() != working_sets.len() {
        return Err("expected one count per selected working set".into());
    }
    for (&count, &working_set) in values.iter().zip(working_sets) {
        if count == 0 || count > 1 << 29 || count % working_set as u64 != 0 {
            return Err(
                "counts must be positive, bounded and divisible by their working sets".into(),
            );
        }
    }
    Ok(values)
}

#[cfg(test)]
fn parse_counts(text: &str) -> Result<[u64; 4], String> {
    parse_counts_for_sets(text, &SETS)?
        .try_into()
        .map_err(|_| "expected four counts".to_string())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut dispatch_counts_text = None;
    let mut lookup_counts_text = None;
    let mut working_sets = SETS.to_vec();
    let mut target = Duration::from_millis(250);
    let mut only = "both".to_string();
    let mut lookup_stride = 16u64;
    while let Some(option) = args.next() {
        let value = args.next().expect("option needs a value");
        match option.as_str() {
            "--dispatch-counts" => dispatch_counts_text = Some(value),
            "--lookup-counts" => lookup_counts_text = Some(value),
            "--working-sets" => working_sets = parse_working_sets(&value).unwrap(),
            "--only" => {
                assert!(matches!(value.as_str(), "both" | "dispatch" | "lookup"));
                only = value;
            }
            "--lookup-stride" => {
                lookup_stride = value.parse().unwrap();
                assert!(
                    (16..=MAX_LOOKUP_STRIDE).contains(&lookup_stride) && lookup_stride % 16 == 0
                );
            }
            "--target-ms" => {
                let millis: u64 = value.parse().unwrap();
                assert!((100..=300).contains(&millis));
                target = Duration::from_millis(millis);
            }
            _ => panic!("unknown option {option}"),
        }
    }
    // Parse counts after all options so --working-sets may appear in any order.
    let dispatch_counts = dispatch_counts_text
        .as_deref()
        .map(|text| parse_counts_for_sets(text, &working_sets).unwrap());
    let lookup_counts = lookup_counts_text
        .as_deref()
        .map(|text| parse_counts_for_sets(text, &working_sets).unwrap());
    let dispatch = if only != "lookup" {
        dispatch_bench(dispatch_counts.as_deref(), &working_sets, target)
    } else {
        vec![]
    };
    let lookup = if only != "dispatch" {
        lookup_bench(lookup_counts.as_deref(), &working_sets, target, lookup_stride)
    } else {
        vec![]
    };
    let list = |values: &[u64]| {
        values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    let dispatch_option = if only != "lookup" {
        format!(" --dispatch-counts {}", list(&dispatch))
    } else {
        String::new()
    };
    let lookup_option = if only != "dispatch" {
        format!(
            " --lookup-counts {} --lookup-stride {lookup_stride}",
            list(&lookup)
        )
    } else {
        String::new()
    };
    let working_sets_option = if working_sets.as_slice() != SETS.as_slice() {
        format!(
            " --working-sets {}",
            working_sets
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    } else {
        String::new()
    };
    println!("matched_arguments=--only {only}{working_sets_option}{dispatch_option}{lookup_option}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_branches_hit_exact_next_block_for_all_working_sets() {
        for working_set in SETS {
            let program = ring(working_set);
            for index in 0..working_set {
                let raw = program.words[index * 4 + 3] & 0x03ff_ffff;
                let displacement = ((raw << 6) as i32 >> 6) as i64 * 4;
                let branch_pc = (CODE + index as u64 * 16 + 12) as i64;
                assert_eq!(
                    branch_pc + displacement,
                    (CODE + ((index + 1) % working_set) as u64 * 16) as i64
                );
            }
        }
    }

    #[test]
    fn fixed_counts_validate_every_working_set_and_overflow_bound() {
        assert_eq!(parse_counts("1,2,32,128").unwrap(), [1, 2, 32, 128]);
        for invalid in [
            "0,2,32,128",
            "1,2,33,128",
            "1,2,32",
            "1,2,32,18446744073709551615",
        ] {
            assert!(parse_counts(invalid).is_err());
        }
    }

    #[test]
    fn extended_working_sets_and_matched_counts_are_bounded_and_exact() {
        let sets = parse_working_sets("256,512,1024,2048,4096").unwrap();
        assert_eq!(sets, [256, 512, 1024, 2048, 4096]);
        assert_eq!(
            parse_counts_for_sets("256,512,1024,2048,4096", &sets).unwrap(),
            [256, 512, 1024, 2048, 4096]
        );
        for invalid in ["", "0", "3", "8192", "256,256", "512,256"] {
            assert!(parse_working_sets(invalid).is_err());
        }
        for invalid in [
            "256,512,1024,2048",
            "256,512,1024,2048,4097",
            "256,512,1024,2048,536875008",
        ] {
            assert!(parse_counts_for_sets(invalid, &sets).is_err());
        }
    }

    #[test]
    fn extended_rings_and_candidate_specific_collision_strides_are_exact() {
        for working_set in [256, 512, 1024, 2048, 4096] {
            let program = ring(working_set);
            assert_eq!(program.words.len(), working_set * 4);
            for index in 0..working_set {
                let raw = program.words[index * 4 + 3] & 0x03ff_ffff;
                let displacement = ((raw << 6) as i32 >> 6) as i64 * 4;
                let branch_pc = (CODE + index as u64 * 16 + 12) as i64;
                assert_eq!(
                    branch_pc + displacement,
                    (CODE + ((index + 1) % working_set) as u64 * 16) as i64
                );
            }
        }
        for (capacity, shift, stride) in [
            (256u64, 10, 262144u64),
            (1024, 12, 4194304),
            (4096, 14, 67108864),
        ] {
            assert!((16..=MAX_LOOKUP_STRIDE).contains(&stride));
            let slot = ((CODE >> 2) ^ (CODE >> shift)) & (capacity - 1);
            for index in 0..MAX_WORKING_SET as u64 {
                let pc = CODE + index * stride;
                assert_eq!(((pc >> 2) ^ (pc >> shift)) & (capacity - 1), slot);
            }
        }
    }

    #[test]
    fn calibration_rounds_up_to_complete_working_set_cycles() {
        assert_eq!(
            iteration_count(Duration::from_millis(100), Duration::from_millis(250), 128),
            2_621_440
        );
        assert_eq!(
            iteration_count(Duration::ZERO, Duration::from_millis(250), 128),
            1 << 29
        );
    }
}
