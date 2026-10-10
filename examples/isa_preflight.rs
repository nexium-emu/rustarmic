//! Compile-only, bounded ISA coverage diagnostic. No guest code is executed.
//!
//! Input is `hex_pc hex_bytes` on stdin, as produced by wonder_code_probe.
//! Bytes are in memory order; each instruction is decoded little-endian.
//! Usage: isa_preflight [--max-instructions DECIMAL] [HEX_SEED ...]
//! The first input row supplies the default seed. Supply additional spans/seeds
//! to cover more routines. RET, indirect branches/calls and UDF end each path.
//! Unsupported non-control instructions are optimistically treated as having
//! fallthrough, so one known gap does not hide later gaps. This is structural
//! coverage, NOT proof of reachability, architectural correctness or gameplay.
//!
//! Only public frontend translation and backend byte emission are used. This
//! example creates no Jit/Engine/CpuContext, executable allocation, guest memory
//! mappings, callbacks, services or GPU objects. Reports go to stdout only.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::{self, BufRead};

use rustarmic::backend::emit_block;
use rustarmic::frontend::translator::{translate_instruction, InstStatus};
use rustarmic::ir::{Block, IrEmitter, Terminal};

const MAX_INPUT_WORDS: usize = 1_048_576;
const DEFAULT_WALK_LIMIT: usize = 65_536;
const EDGE_REPORT_LIMIT: usize = 64;

#[derive(Debug)]
struct Input {
    words: BTreeMap<u64, u32>,
    first_pc: u64,
}

fn hex_u64(value: &str) -> Result<u64, String> {
    let digits = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .unwrap_or(value);
    u64::from_str_radix(digits, 16).map_err(|_| format!("invalid hexadecimal PC {value:?}"))
}

fn parse_rows(reader: impl BufRead) -> Result<Input, String> {
    let mut words = BTreeMap::new();
    let mut first_pc = None;
    for (line_index, line) in reader.lines().enumerate() {
        let line_number = line_index + 1;
        let line = line.map_err(|error| format!("stdin line {line_number}: {error}"))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 2 {
            return Err(format!("line {line_number}: expected hexPC hexbytes"));
        }
        let pc = hex_u64(fields[0])?;
        let bytes = fields[1];
        if pc & 3 != 0 {
            return Err(format!("line {line_number}: unaligned PC {pc:#x}"));
        }
        if bytes.is_empty()
            || bytes.len() % 8 != 0
            || !bytes.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(format!(
                "line {line_number}: bytes must contain complete 4-byte words"
            ));
        }
        if bytes.len() / 8 > MAX_INPUT_WORDS {
            return Err(format!("line {line_number}: input row exceeds word limit"));
        }
        let byte_len = u64::try_from(bytes.len() / 2).map_err(|_| "input length overflow")?;
        pc.checked_add(byte_len)
            .ok_or_else(|| format!("line {line_number}: address range overflows"))?;
        first_pc.get_or_insert(pc);
        for (index, digits) in bytes.as_bytes().chunks_exact(8).enumerate() {
            let mut instruction = [0; 4];
            for (byte, pair) in instruction.iter_mut().zip(digits.chunks_exact(2)) {
                // ASCII hex was validated above, so these slices are UTF-8.
                *byte = u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap();
            }
            let at = pc + index as u64 * 4;
            let word = u32::from_le_bytes(instruction);
            if let Some(previous) = words.insert(at, word) {
                if previous != word {
                    return Err(format!(
                        "line {line_number}: conflicting instruction at {at:#x}"
                    ));
                }
            }
            if words.len() > MAX_INPUT_WORDS {
                return Err(format!(
                    "input exceeds {MAX_INPUT_WORDS} unique instructions"
                ));
            }
        }
    }
    Ok(Input {
        words,
        first_pc: first_pc.ok_or_else(|| "stdin contains no instructions".to_string())?,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Edge {
    from: u64,
    target: Option<u64>,
    reason: &'static str,
}

fn branch_target(pc: u64, immediate: u32, bits: u32) -> u64 {
    let signed = ((immediate as i64) << (64 - bits)) >> (64 - bits);
    pc.wrapping_add_signed(signed << 2)
}

fn successors(pc: u64, word: u32) -> Vec<Edge> {
    let edge = |target, reason| Edge {
        from: pc,
        target,
        reason,
    };
    // UDF imm16: do not wander through zero padding or embedded data.
    if word & 0xffff_0000 == 0 {
        return vec![edge(None, "udf-stop")];
    }
    if word & 0x7c00_0000 == 0x1400_0000 {
        let target = branch_target(pc, word & 0x03ff_ffff, 26);
        let mut edges = vec![edge(Some(target), "direct-branch")];
        if word & 0x8000_0000 != 0 {
            edges.push(edge(pc.checked_add(4), "direct-call-continuation"));
        }
        return edges;
    }
    if word & 0xff00_0000 == 0x5400_0000 || word & 0x7e00_0000 == 0x3400_0000 {
        return vec![
            edge(
                Some(branch_target(pc, (word >> 5) & 0x7ffff, 19)),
                "conditional-taken",
            ),
            edge(pc.checked_add(4), "conditional-fallthrough"),
        ];
    }
    if word & 0x7e00_0000 == 0x3600_0000 {
        return vec![
            edge(
                Some(branch_target(pc, (word >> 5) & 0x3fff, 14)),
                "test-branch-taken",
            ),
            edge(pc.checked_add(4), "test-branch-fallthrough"),
        ];
    }
    match word & 0xffff_fc1f {
        0xd65f_0000 => vec![edge(None, "return-unresolved")],
        0xd61f_0000 => vec![edge(None, "indirect-branch-unresolved")],
        0xd63f_0000 => vec![edge(None, "indirect-call-unresolved")],
        _ => vec![edge(pc.checked_add(4), "fallthrough")],
    }
}

#[derive(Debug)]
struct Walk {
    visited: Vec<u64>,
    unresolved: Vec<Edge>,
    limit_hit: bool,
}

fn walk(input: &Input, seeds: &[u64], limit: usize) -> Result<Walk, String> {
    let mut queue = VecDeque::new();
    let mut scheduled = BTreeSet::new();
    for &seed in seeds {
        if !input.words.contains_key(&seed) {
            return Err(format!(
                "seed {seed:#x} is not an instruction in the supplied spans"
            ));
        }
        if scheduled.insert(seed) {
            queue.push_back(seed);
        }
    }
    let mut visited = Vec::new();
    let mut unresolved = Vec::new();
    while let Some(pc) = queue.pop_front() {
        if visited.len() == limit {
            return Ok(Walk {
                visited,
                unresolved,
                limit_hit: true,
            });
        }
        let word = input.words[&pc];
        visited.push(pc);
        for edge in successors(pc, word) {
            match edge.target {
                Some(target) if input.words.contains_key(&target) => {
                    if scheduled.insert(target) {
                        queue.push_back(target);
                    }
                }
                _ => unresolved.push(edge),
            }
        }
    }
    Ok(Walk {
        visited,
        unresolved,
        limit_hit: false,
    })
}

#[derive(Debug)]
struct Description {
    class: String,
    form: String,
    mnemonic: String,
    display: String,
}

fn describe(word: u32) -> Description {
    match disarm64::decoder::decode(word) {
        Some(opcode) => {
            let operation = format!("{:?}", opcode.operation);
            let mut parts = operation.split('(');
            Description {
                class: parts.next().unwrap_or("unknown").to_string(),
                form: parts.next().unwrap_or("unknown").to_string(),
                mnemonic: format!("{:?}", opcode.mnemonic),
                // Disarm's Display shows PC-relative offsets, not live targets.
                display: opcode.to_string(),
            }
        }
        None => Description {
            class: "decode-none".into(),
            form: "decode-none".into(),
            mnemonic: "unknown".into(),
            display: "unknown".into(),
        },
    }
}

#[derive(Debug)]
struct Failure {
    pc: u64,
    word: u32,
    description: Description,
    error: String,
    count: usize,
}

type Failures = BTreeMap<(&'static str, String), Failure>;

fn record_failure(failures: &mut Failures, stage: &'static str, pc: u64, word: u32, error: String) {
    let description = describe(word);
    let key = (stage, description.form.clone());
    failures
        .entry(key)
        .and_modify(|failure| failure.count += 1)
        .or_insert(Failure {
            pc,
            word,
            description,
            error,
            count: 1,
        });
}

fn preflight(input: &Input, walk: &Walk) -> (Failures, usize, usize) {
    let mut failures = Failures::new();
    let mut emitted_words = 0;
    let mut emitted_bytes = 0;
    for &pc in &walk.visited {
        let word = input.words[&pc];
        if word & 0xffff_0000 == 0 {
            continue;
        }
        let mut block = Block::new(pc);
        let result = translate_instruction(&mut IrEmitter::new(&mut block, pc), word);
        match result {
            Err(error) => {
                record_failure(&mut failures, "frontend", pc, word, error.to_string());
            }
            Ok(status) => {
                block.cycles = 1;
                block.end_pc = pc + 4;
                if status == InstStatus::Continue {
                    block.terminal = Terminal::LinkBlock { next_pc: pc + 4 };
                }
                // Byte emission only: EmittedBlock::code is dropped, never
                // mapped executable, cast to a function pointer or invoked.
                match emit_block(&block) {
                    Ok(emitted) => {
                        emitted_words += 1;
                        emitted_bytes += emitted.code.len();
                    }
                    Err(error) => {
                        record_failure(&mut failures, "codegen", pc, word, error.to_string());
                    }
                }
            }
        }
    }
    (failures, emitted_words, emitted_bytes)
}

// An executable-span inventory is deliberately not a reachability proof.
// Decode and translate known words only; never emit or execute guest code.
fn inventory_frontend(input: &Input, limit: usize) -> (Failures, usize, usize) {
    let mut failures = Failures::new();
    let mut translated = 0;
    let mut skipped = 0;
    for (&pc, &word) in input.words.iter().take(limit) {
        if word & 0xffff_0000 == 0 || disarm64::decoder::decode(word).is_none() {
            skipped += 1;
            continue;
        }
        let mut block = Block::new(pc);
        match translate_instruction(&mut IrEmitter::new(&mut block, pc), word) {
            Ok(_) => translated += 1,
            Err(error) => record_failure(&mut failures, "frontend", pc, word, error.to_string()),
        }
    }
    (failures, translated, skipped)
}

fn run() -> Result<u8, String> {
    let mut seeds = Vec::new();
    let mut limit = DEFAULT_WALK_LIMIT;
    let mut inventory = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("isa_preflight [--inventory-frontend] [--max-instructions DECIMAL] [HEX_SEED ...]");
                println!("stdin: hexPC hexbytes; translation/emission only; no guest execution");
                return Ok(0);
            }
            "--max-instructions" => {
                limit = args
                    .next()
                    .ok_or("missing --max-instructions value")?
                    .parse()
                    .map_err(|_| "invalid decimal instruction limit")?;
                if !(1..=MAX_INPUT_WORDS).contains(&limit) {
                    return Err(format!("instruction limit must be 1..={MAX_INPUT_WORDS}"));
                }
            }
            "--inventory-frontend" => inventory = true,
            _ if arg.starts_with('-') => return Err(format!("unknown option {arg:?}")),
            _ => seeds.push(hex_u64(&arg)?),
        }
    }
    let input = parse_rows(io::stdin().lock())?;
    if inventory {
        if !seeds.is_empty() {
            return Err("frontend inventory does not accept CFG seeds".into());
        }
        let (failures, translated, skipped) = inventory_frontend(&input, limit);
        let mut ranked: Vec<_> = failures.values().collect();
        ranked.sort_by(|a, b| b.count.cmp(&a.count).then(a.pc.cmp(&b.pc)));
        for failure in ranked.iter().take(64) {
            println!(
                "inventory pc={:#x} raw={:#010x} class={} form={} count={} display={:?}",
                failure.pc,
                failure.word,
                failure.description.class,
                failure.description.form,
                failure.count,
                failure.description.display
            );
        }
        println!("inventory-summary supplied_words={} examined={} translated={} skipped={} failure_forms={} reports_omitted={} guest_executed=false",
            input.words.len(), input.words.len().min(limit), translated, skipped,
            failures.len(), failures.len().saturating_sub(64));
        println!("scope=frontend-only; supplied-executable-span-may-contain-data; no-reachability-semantic-or-gameplay-proof");
        return Ok(if failures.is_empty() { 0 } else { 1 });
    }
    if seeds.is_empty() {
        seeds.push(input.first_pc);
    }
    let walk = walk(&input, &seeds, limit)?;
    let (failures, emitted_words, emitted_bytes) = preflight(&input, &walk);
    for ((stage, _), failure) in &failures {
        println!(
            "failure stage={stage} pc={:#x} raw={:#010x} class={} form={} mnemonic={} count={} display={:?} error={:?}",
            failure.pc,
            failure.word,
            failure.description.class,
            failure.description.form,
            failure.description.mnemonic,
            failure.count,
            failure.description.display,
            failure.error,
        );
    }
    for edge in walk.unresolved.iter().take(EDGE_REPORT_LIMIT) {
        let target = edge
            .target
            .map(|pc| format!("{pc:#x}"))
            .unwrap_or_else(|| "unknown".into());
        println!(
            "unresolved from={:#x} target={target} reason={}",
            edge.from, edge.reason
        );
    }
    println!(
        "summary supplied_words={} seeds={} visited={} emitted_words={} emitted_bytes={} failure_forms={} unresolved_edges={} edge_reports_omitted={} limit_hit={} guest_executed=false",
        input.words.len(),
        seeds.len(),
        walk.visited.len(),
        emitted_words,
        emitted_bytes,
        failures.len(),
        walk.unresolved.len(),
        walk.unresolved.len().saturating_sub(EDGE_REPORT_LIMIT),
        walk.limit_hit,
    );
    println!(
        "scope=single-instruction-codegen; unsupported-fallthrough-assumed; indirect-edges-unexplored; no-semantic-or-gameplay-proof"
    );
    Ok(if walk.limit_hit {
        2
    } else if failures.is_empty() {
        0
    } else {
        1
    })
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(code) => std::process::ExitCode::from(code),
        Err(error) => {
            println!("input-error {error}");
            std::process::ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(words: &[(u64, u32)]) -> Input {
        Input {
            words: words.iter().copied().collect(),
            first_pc: words[0].0,
        }
    }

    #[test]
    fn rows_are_little_endian_and_overlap_must_agree() {
        let data = parse_rows("1000 1f2003d5c0035fd6\n1004 c0035fd6\n".as_bytes()).unwrap();
        assert_eq!(data.first_pc, 0x1000);
        assert_eq!(data.words[&0x1000], 0xd503_201f);
        assert_eq!(data.words[&0x1004], 0xd65f_03c0);
        assert!(parse_rows("1000 1f2003d5\n1000 c0035fd6\n".as_bytes()).is_err());
        assert!(parse_rows("1001 1f2003d5\n".as_bytes()).is_err());
        assert!(parse_rows("1000 1f2003\n".as_bytes()).is_err());
        assert!(parse_rows("fffffffffffffffc 1f2003d5\n".as_bytes()).is_err());
    }

    #[test]
    fn cfg_follows_call_and_conditional_paths_but_stops_return_and_padding() {
        let data = input(&[
            (0x1000, 0x9400_0004), // BL 0x1010.
            (0x1004, 0x5400_0040), // B.EQ 0x100c, or fallthrough 0x1008.
            (0x1008, 0xd65f_03c0),
            (0x100c, 0),
            (0x1010, 0x17ff_ffff), // B back to 0x100c.
            (0x1014, 0xd503_201f), // Not reachable after unconditional B.
        ]);
        let result = walk(&data, &[0x1000], 10).unwrap();
        assert_eq!(result.visited.len(), 5);
        assert!(!result.visited.contains(&0x1014));
        assert!(!result.limit_hit);
        assert!(result
            .unresolved
            .iter()
            .any(|edge| edge.reason == "return-unresolved"));
        assert!(result
            .unresolved
            .iter()
            .any(|edge| edge.reason == "udf-stop"));
        assert_eq!(branch_target(0x1000, 0x7ffff, 19), 0x0ffc);
        assert_eq!(branch_target(0x1000, 0x3fff, 14), 0x0ffc);
    }

    #[test]
    fn indirect_calls_are_unresolved_and_walk_is_bounded() {
        let data = input(&[(0x1000, 0xd63f_0100), (0x1004, 0xd503_201f)]);
        let result = walk(&data, &[0x1000], 10).unwrap();
        assert_eq!(result.visited, [0x1000]);
        assert_eq!(result.unresolved[0].reason, "indirect-call-unresolved");
        let data = input(&[(0x1000, 0xd503_201f), (0x1004, 0xd65f_03c0)]);
        assert!(walk(&data, &[0x1000], 1).unwrap().limit_hit);
        assert!(walk(&data, &[0x2000], 10).is_err());
    }

    #[test]
    fn frontend_inventory_counts_known_words_skips_padding_and_obeys_limit() {
        let data = input(&[
            (0x1000, 0xd503_201f),
            (0x1004, 0),
            (0x1008, 1),
            (0x100c, 0xd65f_03c0),
        ]);
        let (failures, translated, skipped) = inventory_frontend(&data, 4);
        assert!(failures.is_empty());
        assert_eq!((translated, skipped), (2, 2));
        let (failures, translated, skipped) = inventory_frontend(&data, 1);
        assert!(failures.is_empty());
        assert_eq!((translated, skipped), (1, 0));
    }

    #[test]
    fn known_words_emit_bytes_without_any_guest_execution() {
        let data = input(&[(0x1000, 0x9100_0400), (0x1004, 0xd65f_03c0)]);
        let result = walk(&data, &[0x1000], 10).unwrap();
        let (failures, count, byte_count) = preflight(&data, &result);
        assert!(failures.is_empty(), "{failures:?}");
        assert_eq!(count, 2);
        assert!(byte_count > 0);
    }
}
