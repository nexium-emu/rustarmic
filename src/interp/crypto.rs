//! AES, SHA-1, SHA-256, SHA-512, SHA-3 helper instructions and CRC32.

const SBOX: [u8; 256] = {
    let mut s = [0u8; 256];
    // Generated from the multiplicative inverse in GF(2^8) plus the affine
    // transform, so no table has to be typed in by hand.
    let mut i = 0;
    while i < 256 {
        let inv = if i == 0 { 0 } else { gf_inv(i as u8) };
        let mut x = inv;
        let mut r = inv;
        let mut k = 0;
        while k < 4 {
            x = x.rotate_left(1);
            r ^= x;
            k += 1;
        }
        s[i] = r ^ 0x63;
        i += 1;
    }
    s
};

const INV_SBOX: [u8; 256] = {
    let mut s = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        s[SBOX[i] as usize] = i as u8;
        i += 1;
    }
    s
};

const fn gf_mul(a: u8, b: u8) -> u8 {
    let mut a = a;
    let mut b = b;
    let mut r = 0u8;
    while b != 0 {
        if b & 1 != 0 {
            r ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1B;
        }
        b >>= 1;
    }
    r
}

const fn gf_inv(a: u8) -> u8 {
    // a^254
    let mut r = 1u8;
    let mut base = a;
    let mut e = 254u32;
    while e != 0 {
        if e & 1 != 0 {
            r = gf_mul(r, base);
        }
        base = gf_mul(base, base);
        e >>= 1;
    }
    r
}

fn bytes(v: u128) -> [u8; 16] {
    v.to_le_bytes()
}

fn from_bytes(b: [u8; 16]) -> u128 {
    u128::from_le_bytes(b)
}

const SHIFT_ROWS: [usize; 16] = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11];
const INV_SHIFT_ROWS: [usize; 16] = [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3];

pub fn aese(d: u128, n: u128) -> u128 {
    let s = bytes(d ^ n);
    let mut o = [0u8; 16];
    for i in 0..16 {
        o[i] = SBOX[s[SHIFT_ROWS[i]] as usize];
    }
    from_bytes(o)
}

pub fn aesd(d: u128, n: u128) -> u128 {
    let s = bytes(d ^ n);
    let mut o = [0u8; 16];
    for i in 0..16 {
        o[i] = INV_SBOX[s[INV_SHIFT_ROWS[i]] as usize];
    }
    from_bytes(o)
}

fn mix(v: u128, m: [u8; 4]) -> u128 {
    let s = bytes(v);
    let mut o = [0u8; 16];
    for c in 0..4 {
        let col = &s[4 * c..4 * c + 4];
        for r in 0..4 {
            o[4 * c + r] = gf_mul(m[0], col[r])
                ^ gf_mul(m[1], col[(r + 1) % 4])
                ^ gf_mul(m[2], col[(r + 2) % 4])
                ^ gf_mul(m[3], col[(r + 3) % 4]);
        }
    }
    from_bytes(o)
}

pub fn aesmc(n: u128) -> u128 {
    mix(n, [2, 3, 1, 1])
}

pub fn aesimc(n: u128) -> u128 {
    mix(n, [14, 11, 13, 9])
}

fn w(v: u128, i: u32) -> u32 {
    (v >> (32 * i)) as u32
}

fn set_w(v: &mut u128, i: u32, x: u32) {
    let m = 0xFFFF_FFFFu128 << (32 * i);
    *v = (*v & !m) | ((x as u128) << (32 * i));
}

fn choose(x: u32, y: u32, z: u32) -> u32 {
    ((y ^ z) & x) ^ z
}
fn majority(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | ((x | y) & z)
}
fn parity(x: u32, y: u32, z: u32) -> u32 {
    x ^ y ^ z
}

/// SHA1C / SHA1P / SHA1M (`kind` 0 / 1 / 2).
pub fn sha1_hash(kind: u32, d: u128, n: u128, m: u128) -> u128 {
    let mut x = d;
    let mut y = w(n, 0);
    for e in 0..4 {
        let (b, c, dd) = (w(x, 1), w(x, 2), w(x, 3));
        let t = match kind {
            0 => choose(b, c, dd),
            1 => parity(b, c, dd),
            _ => majority(b, c, dd),
        };
        y = y
            .wrapping_add(w(x, 0).rotate_left(5))
            .wrapping_add(t)
            .wrapping_add(w(m, e));
        let b2 = w(x, 1).rotate_left(30);
        set_w(&mut x, 1, b2);
        // <Y, X> = ROL(Y : X, 32)
        let top = w(x, 3);
        x = (x << 32) | y as u128;
        y = top;
    }
    x
}

pub fn sha1su0(d: u128, n: u128, m: u128) -> u128 {
    let r = ((n & u64::MAX as u128) << 64) | (d >> 64);
    r ^ d ^ m
}

pub fn sha1su1(d: u128, n: u128) -> u128 {
    let t = d ^ (n >> 32);
    let mut r = 0u128;
    for i in 0..4 {
        set_w(&mut r, i, w(t, i).rotate_left(1));
    }
    let top = w(r, 3) ^ w(t, 0).rotate_left(2);
    set_w(&mut r, 3, top);
    r
}

pub fn sha1h(n: u128) -> u128 {
    w(n, 0).rotate_left(30) as u128
}

fn sigma0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}
fn sigma1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

fn sha256_hash(mut x: u128, mut y: u128, wk: u128, part1: bool) -> u128 {
    for e in 0..4 {
        let chs = choose(w(y, 0), w(y, 1), w(y, 2));
        let maj = majority(w(x, 0), w(x, 1), w(x, 2));
        let t = w(y, 3)
            .wrapping_add(sigma1(w(y, 0)))
            .wrapping_add(chs)
            .wrapping_add(w(wk, e));
        let x3 = t.wrapping_add(w(x, 3));
        set_w(&mut x, 3, x3);
        let y3 = t.wrapping_add(sigma0(w(x, 0))).wrapping_add(maj);
        set_w(&mut y, 3, y3);
        // <Y, X> = ROL(Y : X, 32)
        let xt = w(x, 3);
        let yt = w(y, 3);
        x = (x << 32) | yt as u128;
        y = (y << 32) | xt as u128;
    }
    if part1 { x } else { y }
}

pub fn sha256h(d: u128, n: u128, m: u128) -> u128 {
    sha256_hash(d, n, m, true)
}

pub fn sha256h2(d: u128, n: u128, m: u128) -> u128 {
    sha256_hash(n, d, m, false)
}

pub fn sha256su0(d: u128, n: u128) -> u128 {
    let t = ((n & 0xFFFF_FFFF) << 96) | (d >> 32);
    let mut r = 0u128;
    for e in 0..4 {
        let elt = w(t, e);
        let elt = elt.rotate_right(7) ^ elt.rotate_right(18) ^ (elt >> 3);
        set_w(&mut r, e, elt.wrapping_add(w(d, e)));
    }
    r
}

pub fn sha256su1(d: u128, n: u128, m: u128) -> u128 {
    let t0 = ((m & 0xFFFF_FFFF) << 96) | (n >> 32);
    let mut t1 = m >> 64;
    let mut r = 0u128;
    for e in 0..4 {
        if e == 2 {
            t1 = r & u64::MAX as u128;
        }
        let elt = w(t1, e % 2);
        let elt = elt.rotate_right(17) ^ elt.rotate_right(19) ^ (elt >> 10);
        set_w(&mut r, e, elt.wrapping_add(w(d, e)).wrapping_add(w(t0, e)));
    }
    r
}

fn lo(v: u128) -> u64 {
    v as u64
}
fn hi(v: u128) -> u64 {
    (v >> 64) as u64
}
fn q(h: u64, l: u64) -> u128 {
    ((h as u128) << 64) | l as u128
}

pub fn sha512h(d: u128, n: u128, m: u128) -> u128 {
    let (x, y, wv) = (n, m, d);
    let s1 = |v: u64| v.rotate_right(14) ^ v.rotate_right(18) ^ v.rotate_right(41);
    let mut th = (hi(y) & lo(x)) ^ (!hi(y) & hi(x));
    th = th.wrapping_add(s1(hi(y))).wrapping_add(hi(wv));
    let tmp = th.wrapping_add(lo(y));
    let mut tl = (tmp & hi(y)) ^ (!tmp & lo(x));
    tl = tl.wrapping_add(s1(tmp)).wrapping_add(lo(wv));
    q(th, tl)
}

pub fn sha512h2(d: u128, n: u128, m: u128) -> u128 {
    let (x, y, wv) = (n, m, d);
    let s0 = |v: u64| v.rotate_right(28) ^ v.rotate_right(34) ^ v.rotate_right(39);
    let mut th = (lo(x) & hi(y)) ^ (lo(x) & lo(y)) ^ (hi(y) & lo(y));
    th = th.wrapping_add(s0(lo(y))).wrapping_add(hi(wv));
    let mut tl = (th & lo(y)) ^ (th & hi(y)) ^ (hi(y) & lo(y));
    tl = tl.wrapping_add(s0(th)).wrapping_add(lo(wv));
    q(th, tl)
}

pub fn sha512su0(d: u128, n: u128) -> u128 {
    let s0 = |v: u64| v.rotate_right(1) ^ v.rotate_right(8) ^ (v >> 7);
    let l = lo(d).wrapping_add(s0(hi(d)));
    let h = hi(d).wrapping_add(s0(lo(n)));
    q(h, l)
}

pub fn sha512su1(d: u128, n: u128, m: u128) -> u128 {
    let s1 = |v: u64| v.rotate_right(19) ^ v.rotate_right(61) ^ (v >> 6);
    let h = hi(d).wrapping_add(s1(hi(n))).wrapping_add(hi(m));
    let l = lo(d).wrapping_add(s1(lo(n))).wrapping_add(lo(m));
    q(h, l)
}

pub fn rax1(n: u128, m: u128) -> u128 {
    q(hi(n) ^ hi(m).rotate_left(1), lo(n) ^ lo(m).rotate_left(1))
}

pub fn xar(n: u128, m: u128, imm6: u32) -> u128 {
    let t = n ^ m;
    q(hi(t).rotate_right(imm6), lo(t).rotate_right(imm6))
}

/// CRC32 / CRC32C over the low `bytes` bytes of `data`.
pub fn crc32(acc: u32, data: u64, bytes: u32, castagnoli: bool) -> u32 {
    let poly: u32 = if castagnoli { 0x82F6_3B78 } else { 0xEDB8_8320 };
    let mut crc = acc;
    for i in 0..bytes {
        crc ^= ((data >> (8 * i)) & 0xFF) as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (poly & (crc & 1).wrapping_neg());
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sbox_matches_fips197() {
        assert_eq!(SBOX[0x00], 0x63);
        assert_eq!(SBOX[0x53], 0xED);
        assert_eq!(SBOX[0xFF], 0x16);
        assert_eq!(INV_SBOX[0x63], 0x00);
    }

    #[test]
    fn crc32_check_values() {
        let mut c = 0xFFFF_FFFFu32;
        for b in b"123456789" {
            c = crc32(c, *b as u64, 1, false);
        }
        assert_eq!(!c, 0xCBF4_3926);
        let mut c = 0xFFFF_FFFFu32;
        for b in b"123456789" {
            c = crc32(c, *b as u64, 1, true);
        }
        assert_eq!(!c, 0xE306_9283);
    }
}
