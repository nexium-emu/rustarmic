//! Out-of-line instruction semantics.
//!
//! Instructions the inline translators reject (parts of scalar and vector
//! AdvSIMD, fixed-point to FP conversions, SHA-1 / SHA-512 / SHA-3, dot
//! product) are translated into one `Op::CallHelper` that passes the source
//! registers to [`exec`] and writes the result back.  [`plan`] describes
//! the operands; it accepts exactly the encodings [`exec`] implements.

pub mod crypto;
pub mod fp;
pub mod simd;

use fp::{Fmt, Round, St};

/// A helper operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    /// Whole 128-bit vector register.
    V(u8),
    /// 64-bit general register (31 = XZR).
    X(u8),
    /// 32-bit general register, zero-extended (31 = WZR).
    W(u8),
}

/// Where the helper result goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dst {
    /// Whole vector register (the result carries the zeroed upper part).
    V(u8),
    X(u8),
    W(u8),
}

#[derive(Clone, Copy, Debug)]
pub struct Plan {
    /// Operand slots: 0 = n, 1 = m, 2 = d (old value), 3 = a.
    pub srcs: [Option<Src>; 4],
    pub dst: Dst,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Family {
    ThreeSame(bool),
    ThreeDiff(bool),
    Misc(bool),
    ShiftImm(bool),
    ByElem(bool),
    SameExtra(bool),
    Across,
    ScalarPair,
    ScalarDup,
    Aes,
    Sha3Reg,
    Sha2Reg,
    Sha512,
    Sha512Su0,
    Eor3Bcax,
    Xar,
    Float2Int,
    Float2Fix,
    FloatDp1,
    Crc32,
}

fn bits(x: u32, lsb: u32, len: u32) -> u32 {
    (x >> lsb) & ((1 << len) - 1)
}

fn bit(x: u32, i: u32) -> u32 {
    (x >> i) & 1
}

fn family(insn: u32) -> Option<Family> {
    let b = |lsb, len| bits(insn, lsb, len);
    // CRC32{B,H,W,X} / CRC32C{B,H,W,X}
    if b(29, 2) == 0 && b(21, 8) == 0b1101_0110 && b(13, 3) == 0b010 {
        return Some(Family::Crc32);
    }
    // Conversions between floating-point and integer / fixed point.
    if bit(insn, 30) == 0 && b(24, 5) == 0b11110 {
        if bit(insn, 21) == 1 && b(10, 5) == 0b10000 {
            return Some(Family::FloatDp1);
        }
        if bit(insn, 21) == 1 && b(10, 6) == 0 {
            return Some(Family::Float2Int);
        }
        if bit(insn, 21) == 0 {
            return Some(Family::Float2Fix);
        }
        return None;
    }
    // SHA-512 / SHA-3
    if b(24, 8) == 0xCE {
        if b(21, 3) == 0b011 && b(14, 2) == 0b10 && b(12, 2) == 0 {
            return Some(Family::Sha512);
        }
        if insn & 0xFFFF_FC00 == 0xCEC0_8000 {
            return Some(Family::Sha512Su0);
        }
        if (b(21, 3) == 0 || b(21, 3) == 1) && bit(insn, 15) == 0 {
            return Some(Family::Eor3Bcax);
        }
        if b(21, 3) == 0b100 {
            return Some(Family::Xar);
        }
        return None;
    }
    if b(24, 8) == 0x4E && b(22, 2) == 0 && b(17, 5) == 0b10100 && b(10, 2) == 0b10 {
        return Some(Family::Aes);
    }
    if b(24, 8) == 0x5E && b(22, 2) == 0 {
        if bit(insn, 21) == 0 && bit(insn, 15) == 0 && b(10, 2) == 0 {
            return Some(Family::Sha3Reg);
        }
        if b(17, 5) == 0b10100 && b(10, 2) == 0b10 {
            return Some(Family::Sha2Reg);
        }
    }
    let scalar = match (bit(insn, 31), bit(insn, 30)) {
        (0, 1) if bit(insn, 28) == 1 => true,
        (0, _) if bit(insn, 28) == 0 => false,
        _ => return None,
    };
    if scalar {
        match b(24, 5) {
            0b11110 => {
                if bit(insn, 21) == 1 {
                    if bit(insn, 10) == 1 {
                        return Some(Family::ThreeSame(true));
                    }
                    if b(10, 2) == 0 {
                        return Some(Family::ThreeDiff(true));
                    }
                    if b(10, 2) == 0b10 && b(17, 4) == 0 {
                        return Some(Family::Misc(true));
                    }
                    if b(10, 2) == 0b10 && b(17, 4) == 0b1000 {
                        return Some(Family::ScalarPair);
                    }
                    return None;
                }
                if bit(insn, 15) == 1 && bit(insn, 10) == 1 {
                    return Some(Family::SameExtra(true));
                }
                if b(21, 3) == 0 && bit(insn, 15) == 0 && bit(insn, 10) == 1 {
                    return Some(Family::ScalarDup);
                }
                None
            }
            0b11111 => {
                if bit(insn, 23) == 0 && bit(insn, 10) == 1 {
                    return Some(Family::ShiftImm(true));
                }
                if bit(insn, 10) == 0 {
                    return Some(Family::ByElem(true));
                }
                None
            }
            _ => None,
        }
    } else {
        match b(24, 5) {
            0b01110 => {
                if bit(insn, 21) == 1 {
                    if bit(insn, 10) == 1 {
                        return Some(Family::ThreeSame(false));
                    }
                    if b(10, 2) == 0 {
                        return Some(Family::ThreeDiff(false));
                    }
                    if b(10, 2) == 0b10 && b(17, 4) == 0 {
                        return Some(Family::Misc(false));
                    }
                    if b(10, 2) == 0b10 && b(17, 4) == 0b1000 {
                        return Some(Family::Across);
                    }
                    return None;
                }
                if bit(insn, 15) == 1 && bit(insn, 10) == 1 {
                    return Some(Family::SameExtra(false));
                }
                None
            }
            0b01111 => {
                if bit(insn, 23) == 0 && bit(insn, 10) == 1 && b(19, 4) != 0 {
                    return Some(Family::ShiftImm(false));
                }
                if bit(insn, 10) == 0 {
                    return Some(Family::ByElem(false));
                }
                None
            }
            _ => None,
        }
    }
}

/// Operands and destination for `insn`, or `None` when [`exec`] does not
/// implement it.
pub fn plan(insn: u32) -> Option<Plan> {
    let fam = family(insn)?;
    let rd = bits(insn, 0, 5) as u8;
    let rn = bits(insn, 5, 5) as u8;
    let rm = bits(insn, 16, 5) as u8;
    let ra = bits(insn, 10, 5) as u8;
    let sf = bit(insn, 31) == 1;
    let gpr = |r: u8| if sf { Src::X(r) } else { Src::W(r) };
    let gdst = |r: u8| if sf { Dst::X(r) } else { Dst::W(r) };
    let v3 = [Some(Src::V(rn)), Some(Src::V(rm)), Some(Src::V(rd)), None];
    let p = match fam {
        Family::Crc32 => Plan {
            srcs: [Some(Src::W(rn)), Some(gpr(rm)), None, None],
            dst: Dst::W(rd),
        },
        Family::Float2Int | Family::Float2Fix => {
            if float_to_int(insn) {
                Plan {
                    srcs: [Some(Src::V(rn)), None, None, None],
                    dst: gdst(rd),
                }
            } else {
                Plan {
                    srcs: [Some(gpr(rn)), None, None, None],
                    dst: Dst::V(rd),
                }
            }
        }
        Family::ByElem(_) => {
            let size = bits(insn, 22, 2);
            let fp_op = matches!(
                (bit(insn, 29), bits(insn, 12, 4)),
                (0, 0b0001) | (0, 0b0101) | (0, 0b1001) | (1, 0b1001)
            );
            let rm = if size == 1 && !fp_op { rm & 0xF } else { rm };
            Plan {
                srcs: [Some(Src::V(rn)), Some(Src::V(rm)), Some(Src::V(rd)), None],
                dst: Dst::V(rd),
            }
        }
        Family::FloatDp1 => Plan {
            srcs: [Some(Src::V(rn)), None, None, None],
            dst: Dst::V(rd),
        },
        Family::Eor3Bcax => Plan {
            srcs: [Some(Src::V(rn)), Some(Src::V(rm)), Some(Src::V(rd)), Some(Src::V(ra))],
            dst: Dst::V(rd),
        },
        _ => Plan { srcs: v3, dst: Dst::V(rd) },
    };
    // Accept only what exec implements.
    let zero = [[0u64; 2]; 4];
    exec(insn, &zero, &mut St { fpcr: 0, fpsr: 0 })?;
    Some(p)
}

/// FP -> integer direction of a FLOAT2INT / FLOAT2FIX encoding.
fn float_to_int(insn: u32) -> bool {
    let opcode = bits(insn, 16, 3);
    !matches!(opcode, 0b010 | 0b011)
}

fn v(a: &[u64; 2]) -> u128 {
    (a[0] as u128) | ((a[1] as u128) << 64)
}

/// Executes `insn` on the operands laid out by [`plan`].  Returns `None`
/// for encodings it does not implement (decided from the encoding alone).
pub fn exec(insn: u32, args: &[[u64; 2]; 4], st: &mut St) -> Option<u128> {
    let n = v(&args[0]);
    let m = v(&args[1]);
    let d = v(&args[2]);
    let a = v(&args[3]);
    match family(insn)? {
        Family::ThreeSame(s) => simd::three_same(insn, n, m, d, s, st),
        Family::ThreeDiff(s) => simd::three_different(insn, n, m, d, s, st),
        Family::Misc(s) => simd::two_reg_misc(insn, n, d, s, st),
        Family::ShiftImm(s) => simd::shift_imm(insn, n, d, s, st),
        Family::ByElem(s) => simd::by_element(insn, n, m, d, s, st),
        Family::SameExtra(s) => simd::three_same_extra(insn, n, m, d, s, st),
        Family::Across => simd::across_lanes(insn, n, st),
        Family::ScalarPair => simd::scalar_pairwise(insn, n, st),
        Family::ScalarDup => simd::scalar_dup(insn, n),
        Family::Aes => match bits(insn, 12, 5) {
            0b00100 => Some(crypto::aese(d, n)),
            0b00101 => Some(crypto::aesd(d, n)),
            0b00110 => Some(crypto::aesmc(n)),
            0b00111 => Some(crypto::aesimc(n)),
            _ => None,
        },
        Family::Sha3Reg => match bits(insn, 12, 3) {
            k @ 0..=2 => Some(crypto::sha1_hash(k, d, n, m)),
            3 => Some(crypto::sha1su0(d, n, m)),
            4 => Some(crypto::sha256h(d, n, m)),
            5 => Some(crypto::sha256h2(d, n, m)),
            6 => Some(crypto::sha256su1(d, n, m)),
            _ => None,
        },
        Family::Sha2Reg => match bits(insn, 12, 5) {
            0 => Some(crypto::sha1h(n)),
            1 => Some(crypto::sha1su1(d, n)),
            2 => Some(crypto::sha256su0(d, n)),
            _ => None,
        },
        Family::Sha512 => match bits(insn, 10, 2) {
            0 => Some(crypto::sha512h(d, n, m)),
            1 => Some(crypto::sha512h2(d, n, m)),
            2 => Some(crypto::sha512su1(d, n, m)),
            _ => Some(crypto::rax1(n, m)),
        },
        Family::Sha512Su0 => Some(crypto::sha512su0(d, n)),
        Family::Eor3Bcax => Some(if bits(insn, 21, 3) == 0 {
            n ^ m ^ a
        } else {
            n ^ (m & !a)
        }),
        Family::Xar => Some(crypto::xar(n, m, bits(insn, 10, 6))),
        Family::Crc32 => {
            let sz = bits(insn, 10, 2);
            let sf = bit(insn, 31);
            if (sz == 3) != (sf == 1) {
                return None;
            }
            let c = bit(insn, 12) == 1;
            Some(crypto::crc32(args[0][0] as u32, args[1][0], 1 << sz, c) as u128)
        }
        Family::Float2Int => float2int(insn, args[0][0], st),
        Family::Float2Fix => float2fix(insn, args[0][0], st),
        Family::FloatDp1 => floatdp1(insn, args[0][0], st),
    }
}

fn ftype_fmt(insn: u32) -> Option<Fmt> {
    match bits(insn, 22, 2) {
        0 => Some(Fmt::S),
        1 => Some(Fmt::D),
        _ => None,
    }
}

fn float2int(insn: u32, x: u64, st: &mut St) -> Option<u128> {
    if bit(insn, 29) != 0 {
        return None;
    }
    let f = ftype_fmt(insn)?;
    let width = if bit(insn, 31) == 1 { 64 } else { 32 };
    let rmode = bits(insn, 19, 2);
    let opcode = bits(insn, 16, 3);
    let mode = |r: u32| match r {
        0 => Round::TieEven,
        1 => Round::PosInf,
        2 => Round::NegInf,
        _ => Round::Zero,
    };
    let fx = x & fp::mask(f.bits());
    Some(match (rmode, opcode) {
        (r, 0b000) | (r, 0b001) => fp::to_fixed(f, fx, 0, opcode == 1, width, mode(r), st) as u128,
        (0, 0b100) | (0, 0b101) => fp::to_fixed(f, fx, 0, opcode == 0b101, width, Round::TieAway, st) as u128,
        (0, 0b010) | (0, 0b011) => {
            let r = st.rmode();
            fp::from_fixed(f, x, 0, opcode == 0b011, width, r, st) as u128
        }
        _ => return None,
    })
}

fn float2fix(insn: u32, x: u64, st: &mut St) -> Option<u128> {
    if bit(insn, 29) != 0 {
        return None;
    }
    let f = ftype_fmt(insn)?;
    let sf = bit(insn, 31);
    let width = if sf == 1 { 64 } else { 32 };
    let scale = bits(insn, 10, 6);
    if sf == 0 && scale < 32 {
        return None;
    }
    let fbits = 64 - scale;
    let rmode = bits(insn, 19, 2);
    let opcode = bits(insn, 16, 3);
    let fx = x & fp::mask(f.bits());
    Some(match (rmode, opcode) {
        (0b11, 0b000) | (0b11, 0b001) => {
            fp::to_fixed(f, fx, fbits, opcode == 1, width, Round::Zero, st) as u128
        }
        (0b00, 0b010) | (0b00, 0b011) => {
            let r = st.rmode();
            fp::from_fixed(f, x, fbits, opcode == 0b011, width, r, st) as u128
        }
        _ => return None,
    })
}

/// FP data-processing (1 source): FCVT between half, single and double,
/// and the FRINT* family (single / double).
fn floatdp1(insn: u32, x: u64, st: &mut St) -> Option<u128> {
    if bit(insn, 31) != 0 || bit(insn, 29) != 0 {
        return None;
    }
    let from = match bits(insn, 22, 2) {
        0 => Fmt::S,
        1 => Fmt::D,
        3 => Fmt::H,
        _ => return None,
    };
    let opcode = bits(insn, 15, 6);
    let fx = x & fp::mask(from.bits());
    let r = match opcode {
        0b000100 | 0b000101 | 0b000111 => {
            let to = match opcode & 3 {
                0 => Fmt::S,
                1 => Fmt::D,
                _ => Fmt::H,
            };
            if to == from {
                return None;
            }
            let mode = st.rmode();
            fp::convert(from, to, fx, mode, st)
        }
        0b001000..=0b001111 if from != Fmt::H && opcode != 0b001101 => {
            let mode = match opcode & 7 {
                0 => Round::TieEven,
                1 => Round::PosInf,
                2 => Round::NegInf,
                3 => Round::Zero,
                4 => Round::TieAway,
                _ => st.rmode(),
            };
            fp::round_int(from, fx, mode, opcode == 0b001110, st)
        }
        _ => return None,
    };
    Some(r as u128)
}
