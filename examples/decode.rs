use disarm64::decoder;

fn main() {
    for a in std::env::args().skip(1) {
        let s = a.trim_start_matches("0x");
        match u32::from_str_radix(s, 16) {
            Ok(v) => match decoder::decode(v) {
                Some(op) => println!("{:#010x} {:?}", v, op.operation),
                None => println!("{:#010x} DECODE-FAIL", v),
            },
            Err(_) => println!("{} BAD-HEX", a),
        }
    }
}
