//! CPU-only code-cache publication microbenchmark. Never executes emitted code.
//!
//! Every timed trial installs the same number of distinct 128-byte host blocks
//! into a warmed, otherwise empty 256 MiB cache. Cache creation, reset and the
//! warm-up publication are outside the timer. No guest memory, services or GPU
//! objects are created. Run only with other CPU-intensive work stopped.

use std::hint::black_box;
use std::time::Instant;

use rustarmic::jit::code_cache::CodeCache;

const CACHE_BYTES: usize = 256 * 1024 * 1024;
const BLOCK_BYTES: usize = 128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("usage: code_cache_publish_bench [INSTALLS=2000] [TRIALS=5]".into());
    }
    let installs: usize = args
        .first()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(2_000);
    let trials: usize = args.get(1).map(|v| v.parse()).transpose()?.unwrap_or(5);
    if !(1..=100_000).contains(&installs) || !(1..=100).contains(&trials) {
        return Err("install count must be 1..=100000 and trials 1..=100".into());
    }
    // The bytes are deliberately never called; only the installation path is
    // measured, isolating protection/publication overhead from translation.
    let mut code = [0x90_u8; BLOCK_BYTES];
    code[BLOCK_BYTES - 1] = 0xc3;
    let mut cache = CodeCache::new(CACHE_BYTES)?;
    let mut samples = Vec::with_capacity(trials);
    println!(
        "cache_bytes={CACHE_BYTES} installs={installs} block_bytes={BLOCK_BYTES} trials={trials} guest_executed=false"
    );
    for trial in 0..trials {
        cache.reset()?;
        cache.install(0x1000, 0x1004, 1, &code, &[], 0, true)?;
        let started = Instant::now();
        for index in 0..installs {
            let pc = 0x2000 + index as u64 * 4;
            black_box(cache.install(pc, pc + 4, 1, &code, &[], 0, true)?);
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        black_box(cache.lookup(0x2000 + (installs - 1) as u64 * 4));
        samples.push(elapsed_ms);
        println!("trial={} ms={elapsed_ms:.6}", trial + 1);
    }
    samples.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let median = if samples.len() % 2 == 0 {
        (samples[samples.len() / 2 - 1] + samples[samples.len() / 2]) * 0.5
    } else {
        samples[samples.len() / 2]
    };
    println!(
        "mean_ms={mean:.6} median_ms={median:.6} min_ms={:.6} max_ms={:.6} bytes_per_trial={}",
        samples[0],
        samples[samples.len() - 1],
        installs * BLOCK_BYTES
    );
    Ok(())
}
