const fn multiply(mut a: u8, mut b: u8) -> u8 {
    let mut result = 0;
    let mut bit = 0;
    while bit < 8 {
        result ^= a & 0u8.wrapping_sub(b & 1);
        a = (a << 1) ^ (0x1b & 0u8.wrapping_sub(a >> 7));
        b >>= 1;
        bit += 1;
    }
    result
}

const fn inverse(value: u8) -> u8 {
    let mut result = 1;
    let mut base = value;
    let mut exponent = 254;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = multiply(result, base);
        }
        base = multiply(base, base);
        exponent >>= 1;
    }
    result
}

const fn substitution_table() -> [u8; 256] {
    let mut table = [0; 256];
    let mut index = 0;
    while index < table.len() {
        let value = inverse(index as u8);
        table[index] = value ^ value.rotate_left(1) ^ value.rotate_left(2)
            ^ value.rotate_left(3) ^ value.rotate_left(4) ^ 0x63;
        index += 1;
    }
    table
}

const S_BOX: [u8; 256] = substitution_table();

const fn inverse_substitution_table() -> [u8; 256] {
    let mut table = [0; 256];
    let mut index = 0;
    while index < table.len() {
        table[S_BOX[index] as usize] = index as u8;
        index += 1;
    }
    table
}

const INVERSE_S_BOX: [u8; 256] = inverse_substitution_table();

fn round(input: [u8; 16], decrypt: bool) -> [u8; 16] {
    let table = if decrypt { &INVERSE_S_BOX } else { &S_BOX };
    let mut output = [0; 16];
    for column in 0..4 {
        for row in 0..4 {
            let source_column = if decrypt { (column + 4 - row) & 3 } else { (column + row) & 3 };
            output[column * 4 + row] = table[input[source_column * 4 + row] as usize];
        }
    }
    output
}

fn mix_columns(input: [u8; 16], inverse: bool) -> [u8; 16] {
    let coefficients = if inverse { [14, 11, 13, 9] } else { [2, 3, 1, 1] };
    let mut output = [0; 16];
    for column in 0..4 {
        for row in 0..4 {
            for offset in 0..4 {
                output[column * 4 + row] ^= multiply(
                    input[column * 4 + ((row + offset) & 3)],
                    coefficients[offset],
                );
            }
        }
    }
    output
}

pub(crate) unsafe extern "C" fn transform(
    source: *const [u8; 16],
    destination: *mut [u8; 16],
    mode: u64,
) {
    let input = unsafe { source.read() };
    let output = match mode {
        0 => round(input, false),
        1 => round(input, true),
        2 => mix_columns(input, false),
        3 => mix_columns(input, true),
        _ => unreachable!(),
    };
    unsafe { destination.write(output) };
}
