use disarm64::decoder;
use rustarmic::error::Error;
use rustarmic::frontend::{TranslateOptions, translate_block_into};
use rustarmic::ir::Block;

fn word_at(bytes: &[u8], base: u64, addr: u64) -> Option<u32> {
    let off = addr.checked_sub(base)? as usize;
    if off + 4 > bytes.len() {
        return None;
    }
    Some(u32::from_le_bytes([
        bytes[off],
        bytes[off + 1],
        bytes[off + 2],
        bytes[off + 3],
    ]))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let nro = &args[0];
    let base = 0x8000000000u64;
    let bytes = std::fs::read(nro).expect("read nro");
    let mut block = Block::new(0);
    for pcs in &args[1..] {
        let pc = u64::from_str_radix(pcs.trim_start_matches("0x"), 16).expect("hex pc");
        let mut fetch = |addr: u64| word_at(&bytes, base, addr);
        let r = translate_block_into(&mut block, pc, &mut fetch, TranslateOptions::default());
        match r {
            Ok(()) => println!("{:#x} OK", pc),
            Err(Error::Unsupported { pc: fpc, .. }) | Err(Error::Decode { pc: fpc, .. }) => {
                let w = word_at(&bytes, base, fpc).unwrap_or(0);
                let d = decoder::decode(w)
                    .map(|o| format!("{:?}", o.operation))
                    .unwrap_or_else(|| "DECODE-FAIL".into());
                println!("{:#x} FAIL at {:#x} word={:#010x} {}", pc, fpc, w, d);
            }
            Err(e) => println!("{:#x} ERR {:?}", pc, e),
        }
    }
}
