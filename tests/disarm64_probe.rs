#[allow(dead_code)]
mod common;

use disarm64::decoder;
use disarm64::decoder::Operation;

#[test]
fn decode_smoke_test_instructions() {
    let words: &[(u32, &str)] = &[
        (0xD282_4680, "movz x0, #0x1234"),
        (0xD280_0C80, "movz x0, #100"),
        (0x9100_C800, "add x0, x0, #50"),
        (0xD100_3C00, "sub x0, x0, #15"),
        (0xD420_0000, "brk #0"),
        (0xD2801FE0, "movz x0, #0xFF"),
        (0xAA010002, "orr x2, x0, x1"),
        (0xF9000001, "str x1, [x0]"),
        (0xF9400002, "ldr x2, [x0]"),
    ];

    for &(word, label) in words {
        let opcode = decoder::decode(word)
            .unwrap_or_else(|| panic!("failed to decode {label} ({word:#010x})"));
        let class_name = match &opcode.operation {
            Operation::ADDSUB_IMM(_)   => "ADDSUB_IMM",
            Operation::ADDSUB_SHIFT(_) => "ADDSUB_SHIFT",
            Operation::MOVEWIDE(_)     => "MOVEWIDE",
            Operation::LOG_IMM(_)      => "LOG_IMM",
            Operation::LOG_SHIFT(_)    => "LOG_SHIFT",
            Operation::EXCEPTION(_)    => "EXCEPTION",
            Operation::LDST_POS(_)     => "LDST_POS",
            Operation::LDST_UNSCALED(_)=> "LDST_UNSCALED",
            Operation::BRANCH_IMM(_)   => "BRANCH_IMM",
            Operation::BRANCH_REG(_)   => "BRANCH_REG",
            Operation::CONDBRANCH(_)   => "CONDBRANCH",
            other => {
                eprintln!("{label}: decoded as unexpected class {:?}", other);
                "OTHER"
            }
        };
        eprintln!("{label:<22} {word:#010x} -> mnemonic={:?} class={class_name}",
            opcode.mnemonic);
    }
}


#[test]
fn debug_rev16_correct() {
    for (label, raw) in &[
        ("rev16.16b try1", 0x4E20_1820u32),
        ("rev16.8b try1",  0x0E20_1820u32),
    ] {
        match disarm64::decoder::decode(*raw) {
            Some(op) => eprintln!("{label}: 0x{:08x} -> {:?}", raw, op.operation),
            None => eprintln!("{label}: 0x{:08x} -> DECODE FAIL", raw),
        }
    }
}

#[test]
fn debug_xtn2() {
    let raw = 0x4E21_2820u32;
    let op = disarm64::decoder::decode(raw).expect("decode");
    eprintln!("0x{:08x} -> {:?}", raw, op);
}
