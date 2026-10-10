fn sigma0(value: u32) -> u32 {
    value.rotate_right(7) ^ value.rotate_right(18) ^ (value >> 3)
}

fn sigma1(value: u32) -> u32 {
    value.rotate_right(17) ^ value.rotate_right(19) ^ (value >> 10)
}

fn hash(mut x: [u32; 4], mut y: [u32; 4], words: [u32; 4], part2: bool) -> [u32; 4] {
    for word in words {
        let choose = (y[0] & y[1]) ^ (!y[0] & y[2]);
        let majority = (x[0] & x[1]) ^ (x[0] & x[2]) ^ (x[1] & x[2]);
        let t1 = y[3].wrapping_add(y[0].rotate_right(6) ^ y[0].rotate_right(11)
            ^ y[0].rotate_right(25)).wrapping_add(choose).wrapping_add(word);
        let t2 = (x[0].rotate_right(2) ^ x[0].rotate_right(13)
            ^ x[0].rotate_right(22)).wrapping_add(majority);
        y = [x[3].wrapping_add(t1), y[0], y[1], y[2]];
        x = [t1.wrapping_add(t2), x[0], x[1], x[2]];
    }
    if part2 { y } else { x }
}

fn transform(inputs: [[u32; 4]; 3], mode: u64) -> [u32; 4] {
    let [d, n, m] = inputs;
    match mode {
        0 => hash(d, n, m, false),
        1 => hash(n, d, m, true),
        2 => std::array::from_fn(|i| d[i].wrapping_add(sigma0(if i == 3 { n[0] } else { d[i + 1] }))),
        3 => {
            let next = [n[1], n[2], n[3], m[0]];
            let mut result = [0u32; 4];
            for i in 0..4 {
                let previous = if i < 2 { m[i + 2] } else { result[i - 2] };
                result[i] = d[i].wrapping_add(next[i]).wrapping_add(sigma1(previous));
            }
            result
        }
        _ => unreachable!(),
    }
}

pub(crate) unsafe extern "C" fn portable(
    source: *const [[u32; 4]; 3], destination: *mut [u32; 4], mode: u64,
) {
    let result = transform(unsafe { source.read() }, mode);
    unsafe { destination.write(result) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sha")]
pub(crate) unsafe extern "C" fn accelerated(
    source: *const [[u32; 4]; 3], destination: *mut [u32; 4], mode: u64,
) {
    use std::arch::x86_64::*;
    let [d, n, m] = unsafe { source.read() };
    let d = unsafe { _mm_loadu_si128(d.as_ptr().cast()) };
    let n = unsafe { _mm_loadu_si128(n.as_ptr().cast()) };
    let m = unsafe { _mm_loadu_si128(m.as_ptr().cast()) };
    let result = match mode {
        0 | 1 => {
            let (x, y) = if mode == 0 { (d, n) } else { (n, d) };
            let x = _mm_shuffle_epi32::<0xb1>(x);
            let y = _mm_shuffle_epi32::<0xb1>(y);
            let abef = _mm_unpacklo_epi64(y, x);
            let cdgh = _mm_unpackhi_epi64(y, x);
            let cdgh = _mm_sha256rnds2_epu32(cdgh, abef, m);
            let abef = _mm_sha256rnds2_epu32(abef, cdgh, _mm_shuffle_epi32::<0x0e>(m));
            let output = if mode == 0 { _mm_unpackhi_epi64(abef, cdgh) }
                else { _mm_unpacklo_epi64(abef, cdgh) };
            _mm_shuffle_epi32::<0xb1>(output)
        }
        2 => _mm_sha256msg1_epu32(d, n),
        3 => {
            let next = _mm_or_si128(_mm_srli_si128::<4>(n), _mm_slli_si128::<12>(m));
            _mm_sha256msg2_epu32(_mm_add_epi32(d, next), m)
        }
        _ => unreachable!(),
    };
    unsafe { _mm_storeu_si128(destination.cast(), result) };
}
