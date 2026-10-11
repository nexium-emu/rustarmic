//! Differential fuzzing of the control-flow machinery (block linking, the
//! indirect-branch probe, multiblock translation, compare+branch fusion,
//! budget-capped blocks, page-table fastmem and invalidation).
//!
//! Random but terminating programs with nested loops, direct and indirect
//! calls/returns, forward conditional branches after every flag-setting
//! form, csel/adc flag consumers, memory accesses (plain, pair, Q,
//! writeback, exclusive, LSE) and optionally self-modifying code run under
//! every combination of {link_blocks} x {multiblock} x {page-table fastmem}
//! x {budget 1, 3, 7, 13, unbounded}.  All runs must end in identical
//! register, flag, vector and memory state with identical retired counts.
//! With `--features unicorn` the programs (minus LSE, which Unicorn's CPU
//! model lacks) are additionally checked against Unicorn.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustarmic::jit::context::{PAGE_DIRECT_READ, PAGE_DIRECT_WRITE};
use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};
use std::cell::RefCell;

pub(crate) const MEM_SIZE: usize = 0x40000;
const PAGE_BITS: u8 = 18;
pub(crate) const CODE_BASE: u64 = 0x10000;
pub(crate) const CODE_LIMIT: u64 = 0x20000;
const DATA: u64 = 0x20000;
const DATA_SIZE: u64 = 0x10000;
pub(crate) const STACK_TOP: u64 = 0x3F000;
const BRK_0: u32 = 0xD420_0000;

// Register roles.  Random ALU ops only write x0..x15.
const R_PTR: u32 = 19; // writeback base, reset before each use
const R_DATA: u32 = 20; // data base
const R_LOOP: [u32; 3] = [21, 22, 23]; // main depth 0, main depth 1, functions
const R_CODE: u32 = 24; // code base (SMC stores)
const R_IND: u32 = 16; // indirect call target
const R_JMP: u32 = 17; // indirect jump target

// ---------------------------------------------------------------------------
// Program builder with labels.

#[derive(Clone, Copy)]
enum Fix {
    /// b / bl (imm26)
    B26(u32),
    /// b.cond / cbz / cbnz (imm19 at bit 5)
    B19(u32),
    /// tbz / tbnz (imm14 at bit 5)
    B14(u32),
    /// adr (imm21)
    Adr(u32),
}

struct Asm {
    words: Vec<u32>,
    fixups: Vec<(usize, Fix, usize)>,
    labels: Vec<Option<usize>>,
}

impl Asm {
    fn new() -> Self {
        Self {
            words: Vec::new(),
            fixups: Vec::new(),
            labels: Vec::new(),
        }
    }
    fn label(&mut self) -> usize {
        self.labels.push(None);
        self.labels.len() - 1
    }
    fn bind(&mut self, l: usize) {
        self.labels[l] = Some(self.words.len());
    }
    fn emit(&mut self, w: u32) {
        self.words.push(w);
    }
    fn emit_fix(&mut self, fix: Fix, l: usize) {
        self.fixups.push((self.words.len(), fix, l));
        self.words.push(0);
    }
    fn finish(mut self) -> Vec<u32> {
        for (at, fix, l) in self.fixups.drain(..) {
            let to = self.labels[l].expect("unbound label");
            let d = to as i64 - at as i64;
            self.words[at] = match fix {
                Fix::B26(base) => base | (d as u32 & 0x03FF_FFFF),
                Fix::B19(base) => base | ((d as u32 & 0x7FFFF) << 5),
                Fix::B14(base) => base | ((d as u32 & 0x3FFF) << 5),
                Fix::Adr(rd) => {
                    let b = d * 4;
                    0x1000_0000 | (((b & 3) as u32) << 29) | ((((b >> 2) as u32) & 0x7FFFF) << 5) | rd
                }
            };
        }
        self.words
    }
}

// ---------------------------------------------------------------------------
// Generator.

#[derive(Clone, Copy)]
pub(crate) struct GenOpts {
    pub(crate) lse: bool,
    pub(crate) smc: bool,
}

struct Gen<'a> {
    rng: &'a mut ChaCha8Rng,
    asm: Asm,
    /// Function entry labels; each function starts with the SMC-patchable
    /// `movz x14, #imm`.
    funcs: Vec<usize>,
    opts: GenOpts,
    /// Remaining number of generator steps for the current body.
    budget: usize,
}

fn reg(rng: &mut ChaCha8Rng) -> u32 {
    rng.gen_range(0..16)
}

/// A valid logical-immediate encoding (N, immr, imms) for width `sf`.
fn bitmask_imm(rng: &mut ChaCha8Rng, sf: u32) -> (u32, u32, u32) {
    let sizes: &[u32] = if sf == 1 { &[2, 4, 8, 16, 32, 64] } else { &[2, 4, 8, 16, 32] };
    let e = sizes[rng.gen_range(0..sizes.len())];
    let ones = rng.gen_range(1..e);
    let immr = rng.gen_range(0..e);
    let (n, prefix) = match e {
        64 => (1, 0),
        32 => (0, 0b000000),
        16 => (0, 0b100000),
        8 => (0, 0b110000),
        4 => (0, 0b111000),
        _ => (0, 0b111100),
    };
    (n, immr, prefix | (ones - 1))
}

impl Gen<'_> {
    fn cond(&mut self) -> u32 {
        self.rng.gen_range(0..14)
    }

    fn alu(&mut self) {
        let r = &mut *self.rng;
        let sf: u32 = r.gen_range(0..2);
        let width = if sf == 1 { 64 } else { 32 };
        let (rd, rn, rm, ra) = (reg(r), reg(r), reg(r), reg(r));
        let w = match r.gen_range(0..16) {
            0 => (sf << 31) | (r.gen_range(0..2) << 30) | 0x1100_0000 | (r.gen_range(0..2) << 22)
                | (r.gen_range(0..0x1000) << 10) | (rn << 5) | rd,
            1 => (sf << 31) | (r.gen_range(0..2) << 30) | 0x0B00_0000 | (r.gen_range(0..3) << 22)
                | (rm << 16) | (r.gen_range(0..width) << 10) | (rn << 5) | rd,
            2 => {
                let opc = r.gen_range(0..3); // and/orr/eor
                (sf << 31) | (opc << 29) | 0x0A00_0000 | (r.gen_range(0..4) << 22)
                    | (r.gen_range(0..2) << 21) | (rm << 16) | (r.gen_range(0..width) << 10)
                    | (rn << 5) | rd
            }
            3 => {
                let opc = r.gen_range(0..3);
                let (n, immr, imms) = bitmask_imm(r, sf);
                (sf << 31) | (opc << 29) | 0x1200_0000 | (n << 22) | (immr << 16) | (imms << 10)
                    | (rn << 5) | rd
            }
            4 => (sf << 31) | (r.gen_range(0..2) << 30) | 0x1A00_0000 | (rm << 16) | (rn << 5) | rd,
            5 => {
                let (op, o2) = (r.gen_range(0..2), r.gen_range(0..2));
                (sf << 31) | (op << 30) | 0x1A80_0000 | (rm << 16) | (r.gen_range(0..14) << 12)
                    | (o2 << 10) | (rn << 5) | rd
            }
            6 => (sf << 31) | 0x1B00_0000 | (rm << 16) | (r.gen_range(0..2) << 15) | (ra << 10)
                | (rn << 5) | rd,
            7 => (sf << 31) | 0x1AC0_0800 | (rm << 16) | (r.gen_range(0..2) << 10) | (rn << 5) | rd,
            8 => (sf << 31) | 0x1AC0_2000 | (rm << 16) | (r.gen_range(0..4) << 10) | (rn << 5) | rd,
            9 => {
                let opc = if r.gen_bool(0.5) { 0b10 } else { 0b00 };
                (sf << 31) | (opc << 29) | 0x1300_0000 | (sf << 22) | (r.gen_range(0..width) << 16)
                    | (r.gen_range(0..width) << 10) | (rn << 5) | rd
            }
            10 => (sf << 31) | 0x1380_0000 | (sf << 22) | (rm << 16) | (r.gen_range(0..width) << 10)
                | (rn << 5) | rd,
            11 => (sf << 31) | 0x5AC0_1000 | (r.gen_range(0..2) << 10) | (rn << 5) | rd, // clz/cls
            12 => (sf << 31) | 0x5AC0_0000 | (rn << 5) | rd, // rbit
            13 => {
                let hw = r.gen_range(0..if sf == 1 { 4 } else { 2 });
                let opc = [0b00u32, 0b10, 0b11][r.gen_range(0..3)]; // movn/movz/movk
                (sf << 31) | (opc << 29) | 0x1280_0000 | (hw << 21) | (r.gen_range(0..0x10000) << 5) | rd
            }
            14 => (sf << 31) | 0x5AC0_0800 | (sf << 10) | (rn << 5) | rd, // rev
            _ => 0xD503_201F,
        };
        self.asm.emit(w);
    }

    /// One flag-setting instruction of a random form.
    fn flag_setter(&mut self) {
        let r = &mut *self.rng;
        let sf: u32 = r.gen_range(0..2);
        let width = if sf == 1 { 64 } else { 32 };
        // Rd = zr sometimes (cmp/cmn/tst).
        let rd = if r.gen_bool(0.4) { 31 } else { reg(r) };
        let (rn, rm) = (reg(r), reg(r));
        let w = match r.gen_range(0..8) {
            0 => (sf << 31) | (r.gen_range(0..2) << 30) | 0x3100_0000 | (r.gen_range(0..2) << 22)
                | (r.gen_range(0..0x1000) << 10) | (rn << 5) | rd,
            1 => (sf << 31) | (r.gen_range(0..2) << 30) | 0x2B00_0000 | (r.gen_range(0..3) << 22)
                | (rm << 16) | (r.gen_range(0..width) << 10) | (rn << 5) | rd,
            2 => {
                let (n, immr, imms) = bitmask_imm(r, sf);
                (sf << 31) | (0b11 << 29) | 0x1200_0000 | (n << 22) | (immr << 16) | (imms << 10)
                    | (rn << 5) | rd
            }
            3 => (sf << 31) | (0b11 << 29) | 0x0A00_0000 | (r.gen_range(0..4) << 22)
                | (r.gen_range(0..2) << 21) | (rm << 16) | (r.gen_range(0..width) << 10)
                | (rn << 5) | rd,
            4 => (sf << 31) | (r.gen_range(0..2) << 30) | (1 << 29) | 0x1A00_0000 | (rm << 16)
                | (rn << 5) | rd,
            5 => (sf << 31) | (r.gen_range(0..2) << 30) | (1 << 29) | 0x1A40_0800
                | (r.gen_range(0..32) << 16) | (r.gen_range(0..16) << 12) | (rn << 5)
                | r.gen_range(0..16),
            6 => (sf << 31) | (r.gen_range(0..2) << 30) | (1 << 29) | 0x1A40_0000 | (rm << 16)
                | (r.gen_range(0..16) << 12) | (rn << 5) | r.gen_range(0..16),
            // cmp with a small immediate against a register likely to be
            // small, so equal / near-equal cases are common.
            _ => (sf << 31) | 0x7100_001F | (r.gen_range(0..4) << 10) | (rn << 5),
        };
        self.asm.emit(w);
    }

    fn mem(&mut self) {
        let r = &mut *self.rng;
        let rt = reg(r);
        let load = r.gen_bool(0.5);
        match r.gen_range(0..9) {
            0 => {
                // ldr/str unsigned offset, all sizes, plus sign-extending loads.
                let size = r.gen_range(0..4u32);
                let opc = if load { [1u32, 2, 3][r.gen_range(0..if size == 3 { 1 } else if size == 2 { 2 } else { 3 })] } else { 0 };
                let imm = r.gen_range(0..(DATA_SIZE as u32 >> size).min(4096));
                self.asm.emit((size << 30) | 0x3900_0000 | (opc << 22) | (imm << 10) | (R_DATA << 5) | rt);
            }
            1 => {
                // ldur/stur, unaligned, possibly crossing a page.
                let size = r.gen_range(0..4u32);
                let imm9 = r.gen_range(0..256u32);
                // add x19, x20, #k  or  #k, lsl 12 (k < 15 pages)
                let (sh, k) = if r.gen_bool(0.5) {
                    (1u32, r.gen_range(0..15u32))
                } else {
                    (0, r.gen_range(0..0x1000u32))
                };
                self.asm.emit(0x9100_0000 | (sh << 22) | (k << 10) | (R_DATA << 5) | R_PTR);
                let w = (size << 30) | 0x3800_0000 | ((load as u32) << 22) | (imm9 << 12) | (R_PTR << 5) | rt;
                self.asm.emit(w);
            }
            2 => {
                // pre/post-index writeback on x19.
                let size = r.gen_range(0..4u32);
                let imm9 = r.gen_range(0..512u32);
                self.asm.emit(0x9100_0000 | (1 << 22) | (r.gen_range(1..14u32) << 10) | (R_DATA << 5) | R_PTR);
                let idx = if r.gen_bool(0.5) { 0b11 } else { 0b01 };
                self.asm.emit((size << 30) | 0x3800_0000 | ((load as u32) << 22) | (imm9 << 12) | (idx << 10) | (R_PTR << 5) | rt);
            }
            3 => {
                // register offset with a masked index.
                let size = r.gen_range(0..4u32);
                let idx = reg(r);
                // and x9, x_idx, #0xFFF8 (N=1 immr=61 imms=12 -> 13 ones rotated by 3)
                self.asm.emit(0x9240_0000 | (61 << 16) | (12 << 10) | (idx << 5) | 9);
                let s = r.gen_range(0..2u32);
                let w = (size << 30) | 0x3820_0800 | ((load as u32) << 22) | (9 << 16) | (0b011 << 13) | (s << 12) | (R_DATA << 5) | rt;
                // With S=1 the index is scaled; keep it in range by halving
                // the mask's reach for wide sizes (0xFFF8 << 3 > 64 KiB).
                if s == 1 && size >= 2 {
                    self.asm.emit(0xD341_FD29); // lsr x9, x9, #1
                    self.asm.emit(0xD341_FD29); // lsr x9, x9, #1
                    self.asm.emit(0xD341_FD29); // lsr x9, x9, #1
                }
                self.asm.emit(w);
            }
            4 => {
                // ldp / stp (w or x).
                let wide = r.gen_range(0..2u32);
                let rt2 = loop {
                    let x = reg(r);
                    if x != rt || !load {
                        break x;
                    }
                };
                let imm7 = r.gen_range(0..64u32);
                self.asm.emit((wide << 31) | 0x2900_0000 | ((load as u32) << 22) | (imm7 << 15) | (rt2 << 10) | (R_DATA << 5) | rt);
            }
            5 => {
                // ldr/str q and ldp/stp q (v0..v7).
                let (vt, vt2) = (r.gen_range(0..8u32), r.gen_range(0..8u32));
                if r.gen_bool(0.5) {
                    let imm = r.gen_range(0..(DATA_SIZE as u32 / 16).min(4096));
                    let base = if load { 0x3DC0_0000 } else { 0x3D80_0000 };
                    self.asm.emit(base | (imm << 10) | (R_DATA << 5) | vt);
                } else {
                    let vt2 = if load && vt2 == vt { (vt + 1) & 7 } else { vt2 };
                    let imm7 = r.gen_range(0..64u32);
                    self.asm.emit(0xAD00_0000 | ((load as u32) << 22) | (imm7 << 15) | (vt2 << 10) | (R_DATA << 5) | vt);
                }
            }
            6 => {
                // ldaxr / stlxr pair, maybe with an intervening store.
                let size = r.gen_range(0..4u32);
                let off = r.gen_range(0..256u32) << 4;
                self.asm.emit(0x9100_0000 | (off << 10) | (R_DATA << 5) | R_PTR);
                self.asm.emit((size << 30) | 0x085F_FC00 | (R_PTR << 5) | rt);
                if r.gen_bool(0.3) {
                    self.asm.emit(0x9100_0400 | (rt << 5) | rt); // add rt, rt, #1
                }
                let src = reg(r);
                let status = (src + 1 + r.gen_range(0..15)) % 16;
                self.asm.emit((size << 30) | 0x0800_FC00 | (status << 16) | (R_PTR << 5) | src);
            }
            7 if self.opts.lse => {
                let size = r.gen_range(0..4u32);
                let off = r.gen_range(0..256u32) << 4;
                self.asm.emit(0x9100_0000 | (off << 10) | (R_DATA << 5) | R_PTR);
                let rs = reg(r);
                if r.gen_bool(0.2) {
                    // cas{a,l,al}{b,h,}
                    self.asm.emit((size << 30) | 0x08A0_7C00 | (r.gen_range(0..2) << 22) | (r.gen_range(0..2) << 15) | (rs << 16) | (R_PTR << 5) | rt);
                } else {
                    let opc = r.gen_range(0..9u32);
                    let (o3, opc) = if opc == 8 { (1, 0) } else { (0, opc) };
                    self.asm.emit((size << 30) | 0x3820_0000 | (r.gen_range(0..4) << 22) | (rs << 16) | (o3 << 15) | (opc << 12) | (R_PTR << 5) | rt);
                }
            }
            _ => {
                // ldrsw / ldrsh into x, and strb of a computed value.
                self.asm.emit(0xB980_0000 | (r.gen_range(0..1024) << 10) | (R_DATA << 5) | rt);
            }
        }
    }

    /// Simple straight-line instruction (no control flow).
    fn simple(&mut self) {
        match self.rng.gen_range(0..13) {
            0..=5 => self.alu(),
            6 | 7 => self.mem(),
            8 | 9 => self.flag_setter(),
            10 => self.simd_mem(),
            _ => self.simd_op(),
        }
    }

    /// AdvSIMD structure loads / stores (multiple, single lane, replicate,
    /// with immediate or register post-index), LDNP / STNP and LDTR / STTR.
    fn simd_mem(&mut self) {
        let r = &mut *self.rng;
        // x19 = x20 + k, far enough from both ends of the data region.
        let k = r.gen_range(0x400..0xF000u32) & !0xF;
        let k = k + r.gen_range(0..16u32); // often unaligned
        self.asm.emit(0x9100_0000 | ((k & 0xFFF) << 10) | (R_DATA << 5) | R_PTR); // add x19, x20, #k_lo
        self.asm.emit(0x9140_0000 | ((k >> 12) << 10) | (R_PTR << 5) | R_PTR); // add x19, x19, #k_hi, lsl 12
        let q: u32 = r.gen_range(0..2);
        let l: u32 = r.gen_range(0..2);
        let rt: u32 = r.gen_range(0..32);
        let post = r.gen_bool(0.5);
        // Post-index register: x0..x15 (random, possibly huge: x19 is
        // reset before every use) or 31 for the immediate form.
        let rm: u32 = if r.gen_bool(0.5) { 31 } else { reg(r) };
        let w = match r.gen_range(0..5) {
            0 => {
                let opcode = [0b0000u32, 0b0010, 0b0100, 0b0110, 0b0111, 0b1000, 0b1010][r.gen_range(0..7)];
                let selem1 = matches!(opcode, 0b0010 | 0b0110 | 0b0111 | 0b1010);
                let size = if q == 0 && !selem1 { r.gen_range(0..3u32) } else { r.gen_range(0..4u32) };
                let base = if post { 0x0C80_0000 | (rm << 16) } else { 0x0C00_0000 };
                (q << 30) | base | (l << 22) | (opcode << 12) | (size << 10) | (R_PTR << 5) | rt
            }
            1 => {
                let rr: u32 = r.gen_range(0..2);
                let (opcode, s, size) = if l == 1 && r.gen_bool(0.3) {
                    (0b110 | r.gen_range(0..2u32), 0, r.gen_range(0..4u32))
                } else {
                    match r.gen_range(0..4) {
                        0 => (r.gen_range(0..2u32), r.gen_range(0..2u32), r.gen_range(0..4u32)),
                        1 => (0b010 | r.gen_range(0..2u32), r.gen_range(0..2u32), r.gen_range(0..2u32) << 1),
                        2 => (0b100 | r.gen_range(0..2u32), r.gen_range(0..2u32), 0),
                        _ => (0b100 | r.gen_range(0..2u32), 0, 1),
                    }
                };
                let base = if post { 0x0D80_0000 | (rm << 16) } else { 0x0D00_0000 };
                (q << 30) | base | (l << 22) | (rr << 21) | (opcode << 13) | (s << 12) | (size << 10) | (R_PTR << 5) | rt
            }
            2 => {
                // LDNP / STNP: GPR forms only target x0..x15.
                let (opc, v) = [(0u32, 0u32), (2, 0), (0, 1), (1, 1), (2, 1)][r.gen_range(0..5)];
                let (t, t2) = if v == 0 {
                    let a = reg(r);
                    let b = loop {
                        let b = reg(r);
                        if b != a {
                            break b;
                        }
                    };
                    (a, b)
                } else {
                    let a = r.gen_range(0..32u32);
                    (a, (a + 1 + r.gen_range(0..30u32)) % 32)
                };
                let imm7 = r.gen_range(0..64u32);
                (opc << 30) | 0x2800_0000 | (v << 26) | (l << 22) | (imm7 << 15) | (t2 << 10) | (R_PTR << 5) | t
            }
            3 => {
                // LDTR* / STTR*, signed offset around x19.
                let (size, opc) = [(0u32, 0u32), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3)]
                    [r.gen_range(0..13)];
                let imm9 = r.gen_range(0..512u32);
                (size << 30) | 0x3800_0800 | (opc << 22) | (imm9 << 12) | (R_PTR << 5) | reg(r)
            }
            _ => {
                // The instruction that stopped a game: st1 {vT.s}[idx], [x19]
                0x0D00_8000 | (q << 30) | (r.gen_range(0..2u32) << 12) | (R_PTR << 5) | r.gen_range(0..8u32)
            }
        };
        self.asm.emit(w);
    }

    /// One instruction lowered to the out-of-line helpers (scalar AdvSIMD,
    /// integer vector ops, conversions, CRC32, AES / SHA-1 / SHA-256), as
    /// available on a plain ARMv8.0 + crypto + CRC core.  GPR results go to
    /// x0..x15 only.
    fn simd_op(&mut self) {
        // (fixed, random bits, writes a GPR)
        const RD_RN: u32 = 0x3FF;
        const T: &[(u32, u32, bool)] = &[
            (0x5E20_0400, (1 << 29) | (3 << 22) | (0x1F << 16) | (0x1F << 11) | RD_RN, false), // scalar 3 same
            (0x5E20_0000, (3 << 22) | (0x1F << 16) | (0xF << 12) | RD_RN, false),             // scalar 3 diff
            (0x5E20_0800, (1 << 29) | (3 << 22) | (0x1F << 12) | RD_RN, false),               // scalar misc
            (0x5E30_0800, (1 << 29) | (3 << 22) | (0x1F << 12) | RD_RN, false),               // scalar pairwise
            (0x5E00_0400, (0x1F << 16) | RD_RN, false),                                       // dup scalar
            (0x5F00_0400, (1 << 29) | (0x7F << 16) | (0x1F << 11) | RD_RN, false),            // scalar shift
            (0x5F00_0000, (1 << 29) | (3 << 22) | (0x3F << 16) | (0xF << 12) | (1 << 11) | RD_RN, false), // scalar elem
            (0x0E20_0400, (1 << 30) | (1 << 29) | (3 << 22) | (0x1F << 16) | (0xF << 11) | RD_RN, false), // vector int 3 same
            (0x0F00_0400, (1 << 30) | (1 << 29) | (0x7F << 16) | (0x1F << 11) | RD_RN, false), // vector shift
            (0x0E20_0000, (1 << 30) | (1 << 29) | (3 << 22) | (0x1F << 16) | (0xF << 12) | RD_RN, false), // vector 3 diff
            (0x1E20_0000, (1 << 31) | (1 << 22) | (3 << 19) | (7 << 16) | RD_RN, true),       // float <-> int
            (0x1E00_0000, (1 << 31) | (1 << 22) | (3 << 19) | (7 << 16) | (0x3F << 10) | RD_RN, true), // fixed
            (0x1AC0_4000, (1 << 31) | (0x1F << 16) | (1 << 12) | (3 << 10) | RD_RN, true),    // crc32
            (0x4E28_4800, (3 << 12) | RD_RN, false),                                          // aes
            (0x5E00_0000, (0x1F << 16) | (0x7 << 12) | RD_RN, false),                         // sha1 / sha256 3-reg
            (0x5E28_0800, (3 << 12) | RD_RN, false),                                          // sha 2-reg
        ];
        loop {
            let (fixed, random, gpr_dst) = T[self.rng.gen_range(0..T.len())];
            let mut w = fixed | (self.rng.r#gen::<u32>() & random);
            // Vector templates: integer ops only (inline FP paths are not
            // NaN-exact), and nothing beyond ARMv8.0.
            if fixed == 0x0F00_0400 && ((w >> 19) & 0xF == 0 || (w >> 11) & 0x1F >= 0b11100) {
                continue;
            }
            if fixed == 0x0E20_0000 && (w >> 12) & 0xF == 0b1110 {
                continue; // PMULL: inline for .8b and needs crypto for .1q
            }
            // GPR results (CRC32, FP -> integer) only go to x0..x15; integer
            // -> FP conversions write a vector register.
            if gpr_dst && (fixed == 0x1AC0_4000 || !matches!((w >> 16) & 7, 0b010 | 0b011)) {
                w = (w & !31) | reg(self.rng);
            }
            // SQRDMLAH / SQRDMLSH (by element) are ARMv8.1.
            if fixed == 0x5F00_0000 && w & (1 << 29) != 0 && matches!((w >> 12) & 0xF, 0b1101 | 0b1111) {
                continue;
            }
            // Only encodings the helpers implement (which are exactly the
            // allocated ones); sha 3-reg opcode 7 is unallocated.
            if rustarmic::interp::plan(w).is_none() {
                continue;
            }
            // SHA1H & co. need size 00; the template already fixes it.
            self.asm.emit(w);
            return;
        }
    }

    /// Forward conditional skip over a few simple instructions.
    fn forward_branch(&mut self) {
        let l = self.asm.label();
        let r = &mut *self.rng;
        match r.gen_range(0..4) {
            0 | 1 => {
                // flag setter immediately followed by b.cond (fusion
                // candidate) or separated by non-flag work.
                self.flag_setter();
                if self.rng.gen_bool(0.3) {
                    self.alu_no_flags();
                }
                let c = self.cond();
                self.asm.emit_fix(Fix::B19(0x5400_0000 | c), l);
            }
            2 => {
                let sf: u32 = r.gen_range(0..2);
                let op: u32 = r.gen_range(0..2);
                let rt = reg(r);
                self.asm.emit_fix(Fix::B19((sf << 31) | 0x3400_0000 | (op << 24) | rt), l);
            }
            _ => {
                let bit: u32 = r.gen_range(0..64);
                let op: u32 = r.gen_range(0..2);
                let rt = reg(r);
                self.asm.emit_fix(
                    Fix::B14(((bit >> 5) << 31) | 0x3600_0000 | (op << 24) | ((bit & 31) << 19) | rt),
                    l,
                );
            }
        }
        let n = self.rng.gen_range(1..5);
        for _ in 0..n {
            self.simple();
        }
        self.asm.bind(l);
        // Flag consumers that are not branches.
        if self.rng.gen_bool(0.3) {
            self.alu();
        }
    }

    /// ALU op that never writes NZCV (keeps flags for a later b.cond).
    fn alu_no_flags(&mut self) {
        let r = &mut *self.rng;
        let (rd, rn) = (reg(r), reg(r));
        self.asm.emit(0x9100_0000 | (r.gen_range(0..0x1000) << 10) | (rn << 5) | rd);
    }

    fn seq(&mut self, depth: usize, len: usize, in_func: bool) {
        for _ in 0..len {
            if self.budget == 0 {
                return;
            }
            self.budget -= 1;
            let pick = self.rng.gen_range(0..100);
            match pick {
                0..=39 => self.simple(),
                40..=64 => self.forward_branch(),
                65..=72 if !in_func && depth < 2 || in_func && depth < 1 => {
                    // counted loop
                    let counter = if in_func { R_LOOP[2] } else { R_LOOP[depth] };
                    let n = self.rng.gen_range(1..6u32);
                    self.asm.emit(0xD280_0000 | (n << 5) | counter); // movz
                    let top = self.asm.label();
                    self.asm.bind(top);
                    let body = self.rng.gen_range(1..8);
                    self.seq(depth + 1, body, in_func);
                    self.asm.emit(0xF100_0400 | (counter << 5) | counter); // subs xC, xC, #1
                    self.asm.emit_fix(Fix::B19(0x5400_0001), top); // b.ne top
                }
                73..=80 if !in_func && !self.funcs.is_empty() => {
                    let f = self.funcs[self.rng.gen_range(0..self.funcs.len())];
                    if self.rng.gen_bool(0.5) {
                        self.asm.emit_fix(Fix::B26(0x9400_0000), f); // bl
                    } else {
                        self.asm.emit_fix(Fix::Adr(R_IND), f);
                        self.asm.emit(0xD63F_0000 | (R_IND << 5)); // blr
                    }
                }
                81..=85 => {
                    // indirect forward jump over junk
                    let l = self.asm.label();
                    self.asm.emit_fix(Fix::Adr(R_JMP), l);
                    self.asm.emit(0xD61F_0000 | (R_JMP << 5)); // br
                    self.asm.emit(0xD280_0000 | 0xDEAD << 5); // never executed
                    self.asm.bind(l);
                }
                86..=89 => {
                    // unconditional forward b (multiblock follows these)
                    let l = self.asm.label();
                    self.asm.emit_fix(Fix::B26(0x1400_0000), l);
                    self.asm.emit(0xD280_0000 | 0xBEEF << 5);
                    self.asm.bind(l);
                }
                90..=93 if !in_func && self.opts.smc && !self.funcs.is_empty() => {
                    // Self-modifying code: rewrite the first instruction
                    // (movz x14, #imm) of a function, then leave the block
                    // through an indirect branch so the store is never
                    // followed by stale code in the same host block.
                    let f = self.funcs[self.rng.gen_range(0..self.funcs.len())];
                    let imm = self.rng.gen_range(0..0x10000u32);
                    let word = 0xD280_0000 | (imm << 5) | 14;
                    self.asm.emit(0x5280_0000 | ((word & 0xFFFF) << 5) | 13); // movz w13, lo
                    self.asm.emit(0x72A0_0000 | ((word >> 16) << 5) | 13); // movk w13, hi, lsl 16
                    self.asm.emit_fix(Fix::Adr(R_PTR), f);
                    self.asm.emit(0xB900_0000 | (R_PTR << 5) | 13); // str w13, [x19]
                    let l = self.asm.label();
                    self.asm.emit_fix(Fix::Adr(R_JMP), l);
                    self.asm.emit(0xD61F_0000 | (R_JMP << 5));
                    self.asm.bind(l);
                }
                _ => self.simple(),
            }
        }
    }
}

pub(crate) struct Program {
    pub(crate) words: Vec<u32>,
    pub(crate) init_x: [u64; 31],
    pub(crate) init_v: [[u64; 2]; 8],
    pub(crate) init_nzcv: u8,
    pub(crate) data: Vec<u8>,
}

pub(crate) fn gen_program(seed: u64, opts: GenOpts) -> Program {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let nfuncs = rng.gen_range(0..4);
    let mut g = Gen {
        rng: &mut rng,
        asm: Asm::new(),
        funcs: Vec::new(),
        opts,
        budget: 0,
    };
    for _ in 0..nfuncs {
        let l = g.asm.label();
        g.funcs.push(l);
    }
    g.budget = g.rng.gen_range(10..60);
    let len = g.budget;
    g.seq(0, len, false);
    g.asm.emit(BRK_0);
    let funcs = g.funcs.clone();
    for f in funcs {
        g.asm.bind(f);
        let imm = g.rng.gen_range(0..0x10000u32);
        g.asm.emit(0xD280_0000 | (imm << 5) | 14); // movz x14, #imm (SMC slot)
        g.budget = g.rng.gen_range(2..15);
        let len = g.budget;
        g.seq(0, len, true);
        g.asm.emit(0xD65F_03C0); // ret
    }
    let words = g.asm.finish();

    let mut init_x = [0u64; 31];
    for (i, x) in init_x.iter_mut().enumerate() {
        *x = match rng.gen_range(0..4) {
            0 => rng.gen_range(0..8),
            1 => rng.r#gen::<u32>() as u64,
            2 => rng.r#gen::<u64>(),
            _ => (rng.r#gen::<i8>() as i64) as u64,
        };
        let _ = i;
    }
    init_x[R_PTR as usize] = DATA;
    init_x[R_DATA as usize] = DATA;
    init_x[R_CODE as usize] = CODE_BASE;
    for r in R_LOOP {
        init_x[r as usize] = 0;
    }
    init_x[30] = 0;
    let mut init_v = [[0u64; 2]; 8];
    for v in &mut init_v {
        *v = [rng.r#gen(), rng.r#gen()];
    }
    let mut data = vec![0u8; DATA_SIZE as usize];
    rng.fill(&mut data[..]);
    Program {
        words,
        init_x,
        init_v,
        init_nzcv: rng.gen_range(0..16),
        data,
    }
}

// ---------------------------------------------------------------------------
// Rustarmic runner.

struct Guest {
    mem: Vec<u8>,
    invalidations: Vec<(u64, u64)>,
}

thread_local! {
    static GUEST: RefCell<Option<*mut Guest>> = const { RefCell::new(None) };
}

fn guest() -> &'static mut Guest {
    GUEST.with(|g| unsafe { &mut *g.borrow().expect("no guest") })
}

fn fault(ctx: *mut CpuContext, addr: u64, size: u8, write: bool) {
    unsafe {
        let ctx = &mut *ctx;
        ctx.mem_fault = 1;
        ctx.mem_fault_access = write as u8;
        ctx.mem_fault_size = size;
        ctx.mem_fault_addr = addr;
        ctx.mem_fault_pc = ctx.pc;
    }
}

fn in_range(addr: u64, size: u8) -> bool {
    addr >= 0x1000 && addr.checked_add(size as u64).is_some_and(|e| e <= MEM_SIZE as u64)
}

unsafe extern "C" fn hk_read(ctx: *mut CpuContext, addr: u64, size: u8) {
    if !in_range(addr, size) {
        fault(ctx, addr, size, false);
        unsafe { (*ctx).io_value = [0; 2] };
        return;
    }
    let g = guest();
    let mut buf = [0u8; 16];
    buf[..size as usize].copy_from_slice(&g.mem[addr as usize..][..size as usize]);
    unsafe {
        (*ctx).io_value = [
            u64::from_le_bytes(buf[..8].try_into().unwrap()),
            u64::from_le_bytes(buf[8..].try_into().unwrap()),
        ];
    }
}

unsafe extern "C" fn hk_write(ctx: *mut CpuContext, addr: u64, size: u8) {
    if !in_range(addr, size) {
        fault(ctx, addr, size, true);
        return;
    }
    let g = guest();
    let io = unsafe { (*ctx).io_value };
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&io[0].to_le_bytes());
    buf[8..].copy_from_slice(&io[1].to_le_bytes());
    g.mem[addr as usize..][..size as usize].copy_from_slice(&buf[..size as usize]);
    if addr < CODE_LIMIT && addr + size as u64 > CODE_BASE {
        g.invalidations.push((addr, size as u64));
        unsafe { (*ctx).should_halt = 1 };
    }
}

unsafe extern "C" fn hk_atomic(ctx: *mut CpuContext, addr: u64, size: u8, op: u8) {
    if !in_range(addr, size) {
        fault(ctx, addr, size, true);
        return;
    }
    let g = guest();
    unsafe {
        rustarmic::jit::atomic::context_host_atomic(ctx, g.mem.as_mut_ptr().add(addr as usize), size, op)
    };
    if addr < CODE_LIMIT && addr + size as u64 > CODE_BASE {
        g.invalidations.push((addr, size as u64));
        unsafe { (*ctx).should_halt = 1 };
    }
}

struct Fetch;
impl Memory for Fetch {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        if !(CODE_BASE..CODE_LIMIT).contains(&addr) {
            return None;
        }
        let g = guest();
        Some(u32::from_le_bytes(g.mem[addr as usize..][..4].try_into().unwrap()))
    }
}

#[derive(Clone, Copy, Debug)]
struct Config {
    link: bool,
    multiblock: bool,
    pagetable: bool,
    budget: u64,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct Outcome {
    x: [u64; 31],
    sp: u64,
    pc: u64,
    nzcv: u8,
    v: [[u64; 2]; 8],
    reason: String,
    retired: u64,
    mem_hash: u64,
    data: Vec<u8>,
    invalidations: usize,
}

fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    h
}

pub(crate) fn image(p: &Program) -> Vec<u8> {
    let mut mem = vec![0u8; MEM_SIZE];
    for (i, w) in p.words.iter().enumerate() {
        mem[CODE_BASE as usize + 4 * i..][..4].copy_from_slice(&w.to_le_bytes());
    }
    mem[DATA as usize..][..DATA_SIZE as usize].copy_from_slice(&p.data);
    mem
}

const MAX_STEPS: u64 = 2_000_000;

fn run_rustarmic(p: &Program, cfg: Config) -> Outcome {
    let mut g = Box::new(Guest {
        mem: image(p),
        invalidations: Vec::new(),
    });
    GUEST.with(|s| *s.borrow_mut() = Some(&mut *g as *mut Guest));
    let mut table = vec![PAGE_DIRECT_READ | PAGE_DIRECT_WRITE; MEM_SIZE >> 12];
    table[0] = 0;
    for page in (CODE_BASE >> 12)..(CODE_LIMIT >> 12) {
        table[page as usize] = PAGE_DIRECT_READ;
    }
    let mut config = JitConfig {
        code_cache_bytes: 4 << 20,
        link_blocks: cfg.link,
        ..JitConfig::default()
    };
    config.translate.multiblock = cfg.multiblock;
    if cfg.pagetable {
        config.fastmem_page_bits = Some(PAGE_BITS);
    }
    let mut jit = Jit::new(config).unwrap();
    let mut ctx = CpuContext::default();
    ctx.mem_read = hk_read;
    ctx.mem_write = hk_write;
    ctx.mem_atomic = hk_atomic;
    ctx.mem_base = g.mem.as_mut_ptr();
    ctx.page_table = table.as_ptr();
    ctx.pc = CODE_BASE;
    ctx.sp = STACK_TOP;
    ctx.x = p.init_x;
    ctx.nzcv = p.init_nzcv;
    for i in 0..8 {
        ctx.v[i] = p.init_v[i];
    }
    let mut retired = 0u64;
    let mut invalidations = 0usize;
    let reason = loop {
        let pc = ctx.pc;
        let out = match jit.run_bounded(&mut ctx, &mut Fetch, cfg.budget) {
            Ok(out) => out,
            Err(e) => {
                let at = ((pc - CODE_BASE) / 4) as usize;
                panic!(
                    "{cfg:?}: run_bounded failed at pc {pc:#x}: {e:?}; code from there: {:08x?}",
                    &p.words[at.min(p.words.len())..(at + 24).min(p.words.len())]
                )
            }
        };
        retired += out.retired;
        let pending = std::mem::take(&mut g.invalidations);
        for (a, l) in &pending {
            jit.invalidate_range(*a, *l);
        }
        invalidations += pending.len();
        match out.reason {
            ExitReason::BudgetExhausted if retired < MAX_STEPS => continue,
            ExitReason::Stopped if !pending.is_empty() && retired < MAX_STEPS => continue,
            other => break format!("{other:?} {:?}", out.stop),
        }
    };
    GUEST.with(|s| *s.borrow_mut() = None);
    let mut v = [[0u64; 2]; 8];
    v.copy_from_slice(&ctx.v[..8]);
    Outcome {
        x: ctx.x,
        sp: ctx.sp,
        pc: ctx.pc,
        nzcv: ctx.nzcv & 0xF,
        v,
        reason,
        retired,
        mem_hash: fnv(&g.mem),
        data: g.mem[DATA as usize..][..DATA_SIZE as usize].to_vec(),
        invalidations,
    }
}

fn configs() -> Vec<Config> {
    let mut out = Vec::new();
    for link in [false, true] {
        for multiblock in [false, true] {
            for pagetable in [false, true] {
                for budget in [1, 3, 7, 13, u64::MAX] {
                    out.push(Config {
                        link,
                        multiblock,
                        pagetable,
                        budget,
                    });
                }
            }
        }
    }
    out
}

fn diff(a: &Outcome, b: &Outcome) -> String {
    let mut s = String::new();
    for i in 0..31 {
        if a.x[i] != b.x[i] {
            s += &format!(" x{i}: {:#x} vs {:#x};", a.x[i], b.x[i]);
        }
    }
    if a.nzcv != b.nzcv {
        s += &format!(" nzcv {:04b} vs {:04b};", a.nzcv, b.nzcv);
    }
    if a.pc != b.pc {
        s += &format!(" pc {:#x} vs {:#x};", a.pc, b.pc);
    }
    if a.retired != b.retired {
        s += &format!(" retired {} vs {};", a.retired, b.retired);
    }
    if a.reason != b.reason {
        s += &format!(" reason {} vs {};", a.reason, b.reason);
    }
    if a.v != b.v {
        s += " v regs differ;";
    }
    if a.mem_hash != b.mem_hash {
        let first = a.data.iter().zip(&b.data).position(|(x, y)| x != y);
        s += &format!(" memory differs (first data byte {first:?});");
    }
    s
}

fn self_differential(seed: u64, opts: GenOpts) -> Program {
    let p = gen_program(seed, opts);
    let cfgs = configs();
    let base = run_rustarmic(&p, cfgs[cfgs.len() - 1]);
    assert!(base.reason.starts_with("Brk(0)"), "seed {seed}: baseline did not reach brk: {}", base.reason);
    for cfg in &cfgs {
        let o = run_rustarmic(&p, *cfg);
        if o != base {
            panic!(
                "seed {seed} opts lse={} smc={}: {cfg:?} differs from baseline:{}\nprogram: {:08x?}",
                opts.lse,
                opts.smc,
                diff(&base, &o),
                p.words
            );
        }
    }
    p
}

fn seeds() -> u64 {
    std::env::var("RUSTARMIC_CF_FUZZ_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100)
}

#[test]
fn control_flow_configs_agree() {
    for seed in 0..seeds() {
        self_differential(seed, GenOpts { lse: true, smc: false });
    }
}

#[test]
fn control_flow_configs_agree_with_smc() {
    let mut saw_invalidation = false;
    for seed in 1000..1000 + seeds() {
        let p = gen_program(seed, GenOpts { lse: true, smc: true });
        let o = run_rustarmic(&p, configs()[0]);
        saw_invalidation |= o.invalidations > 0;
        self_differential(seed, GenOpts { lse: true, smc: true });
    }
    assert!(saw_invalidation, "no program exercised self-modifying code");
}

// ---------------------------------------------------------------------------
// Unicorn reference.

#[cfg(feature = "unicorn")]
mod unicorn_ref {
    use super::*;
    use unicorn_engine::{Arch, Mode, Prot, RegisterARM64, Unicorn};

    fn xreg(i: usize) -> RegisterARM64 {
        use RegisterARM64::*;
        [
            X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18,
            X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30,
        ][i]
    }
    fn qreg(i: usize) -> RegisterARM64 {
        use RegisterARM64::*;
        [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7][i]
    }

    pub fn run(p: &Program) -> (Outcome, u64) {
        let mut emu = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
        emu.mem_map(0x1000, MEM_SIZE as u64 - 0x1000, Prot::ALL).unwrap();
        let img = image(p);
        emu.mem_write(0x1000, &img[0x1000..]).unwrap();
        let cpacr = emu.reg_read(RegisterARM64::CPACR_EL1).unwrap_or(0);
        let _ = emu.reg_write(RegisterARM64::CPACR_EL1, cpacr | (0b11 << 20));
        for i in 0..31 {
            emu.reg_write(xreg(i), p.init_x[i]).unwrap();
        }
        emu.reg_write(RegisterARM64::SP, STACK_TOP).unwrap();
        emu.reg_write(RegisterARM64::NZCV, (p.init_nzcv as u64) << 28).unwrap();
        for i in 0..8 {
            let mut b = [0u8; 16];
            b[..8].copy_from_slice(&p.init_v[i][0].to_le_bytes());
            b[8..].copy_from_slice(&p.init_v[i][1].to_le_bytes());
            emu.reg_write_long(qreg(i), &b).unwrap();
        }
        let count = std::rc::Rc::new(std::cell::Cell::new(0u64));
        let c2 = count.clone();
        emu.add_code_hook(CODE_BASE, CODE_LIMIT, move |_, _, _| c2.set(c2.get() + 1))
            .unwrap();
        // The main body's final BRK is the first BRK word in the image.
        let brk_at = CODE_BASE + 4 * p.words.iter().position(|&w| w == BRK_0).unwrap() as u64;
        let res = emu.emu_start(CODE_BASE, brk_at, 0, MAX_STEPS as usize);
        let mut x = [0u64; 31];
        for i in 0..31 {
            x[i] = emu.reg_read(xreg(i)).unwrap();
        }
        let mut v = [[0u64; 2]; 8];
        for i in 0..8 {
            let b = emu.reg_read_long(qreg(i)).unwrap();
            v[i] = [
                u64::from_le_bytes(b[..8].try_into().unwrap()),
                u64::from_le_bytes(b[8..16].try_into().unwrap()),
            ];
        }
        let mut mem = vec![0u8; MEM_SIZE];
        emu.mem_read(0x1000, &mut mem[0x1000..]).unwrap();
        let pc = emu.reg_read(RegisterARM64::PC).unwrap();
        let out = Outcome {
            x,
            sp: emu.reg_read(RegisterARM64::SP).unwrap(),
            pc,
            nzcv: ((emu.reg_read(RegisterARM64::NZCV).unwrap() >> 28) & 0xF) as u8,
            v,
            reason: format!("{res:?}"),
            retired: count.get(),
            mem_hash: fnv(&mem),
            data: mem[DATA as usize..][..DATA_SIZE as usize].to_vec(),
            invalidations: 0,
        };
        (out, brk_at)
    }

    #[test]
    fn control_flow_matches_unicorn() {
        for seed in 0..seeds() {
            for smc in [false, true] {
                let p = gen_program(seed + 5000, GenOpts { lse: false, smc });
                let (uni, brk_at) = run(&p);
                assert!(uni.reason.contains("Ok"), "seed {seed}: unicorn {:?}", uni.reason);
                assert_eq!(uni.pc, brk_at);
                let jit = run_rustarmic(
                    &p,
                    Config {
                        link: true,
                        multiblock: true,
                        pagetable: true,
                        budget: 7,
                    },
                );
                let mut jit_cmp = jit.clone();
                // Both report the BRK's own PC; Rustarmic counts it retired.
                jit_cmp.retired = jit.retired - 1;
                jit_cmp.reason = uni.reason.clone();
                jit_cmp.invalidations = 0;
                if jit_cmp != uni {
                    panic!(
                        "seed {} smc={smc}: rustarmic vs unicorn:{}\nprogram: {:08x?}",
                        seed + 5000,
                        diff(&uni, &jit_cmp),
                        p.words
                    );
                }
                // And the full configuration matrix agrees with itself.
                let _ = self_differential(seed + 5000, GenOpts { lse: false, smc });
            }
        }
    }

    /// Investigation helper: single-steps Unicorn and Rustarmic side by side
    /// and reports the first instruction after which registers or flags
    /// differ.  `RUSTARMIC_CF_SEED=<seed> [RUSTARMIC_CF_SMC=1]`.
    #[test]
    #[ignore = "investigation helper"]
    fn trace_first_divergence() {
        let seed: u64 = std::env::var("RUSTARMIC_CF_SEED").unwrap().parse().unwrap();
        let smc = std::env::var("RUSTARMIC_CF_SMC").is_ok();
        let p = gen_program(seed, GenOpts { lse: false, smc });
        let mut emu = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
        emu.mem_map(0x1000, MEM_SIZE as u64 - 0x1000, Prot::ALL).unwrap();
        let img = image(&p);
        emu.mem_write(0x1000, &img[0x1000..]).unwrap();
        let cpacr = emu.reg_read(RegisterARM64::CPACR_EL1).unwrap_or(0);
        let _ = emu.reg_write(RegisterARM64::CPACR_EL1, cpacr | (0b11 << 20));
        for i in 0..31 {
            emu.reg_write(xreg(i), p.init_x[i]).unwrap();
        }
        emu.reg_write(RegisterARM64::SP, STACK_TOP).unwrap();
        emu.reg_write(RegisterARM64::NZCV, (p.init_nzcv as u64) << 28).unwrap();
        for i in 0..8 {
            let mut b = [0u8; 16];
            b[..8].copy_from_slice(&p.init_v[i][0].to_le_bytes());
            b[8..].copy_from_slice(&p.init_v[i][1].to_le_bytes());
            emu.reg_write_long(qreg(i), &b).unwrap();
        }

        let mut g = Box::new(Guest { mem: image(&p), invalidations: Vec::new() });
        GUEST.with(|s| *s.borrow_mut() = Some(&mut *g as *mut Guest));
        let mut config = JitConfig { code_cache_bytes: 4 << 20, ..JitConfig::default() };
        config.translate.multiblock = false;
        let mut jit = Jit::new(config).unwrap();
        let mut ctx = CpuContext::default();
        ctx.mem_read = hk_read;
        ctx.mem_write = hk_write;
        ctx.mem_atomic = hk_atomic;
        ctx.pc = CODE_BASE;
        ctx.sp = STACK_TOP;
        ctx.x = p.init_x;
        ctx.nzcv = p.init_nzcv;
        for i in 0..8 {
            ctx.v[i] = p.init_v[i];
        }
        for step in 0..MAX_STEPS {
            let pc = ctx.pc;
            let word = p.words[((pc - CODE_BASE) / 4) as usize];
            if word == BRK_0 {
                eprintln!("reached brk after {step} steps without divergence");
                return;
            }
            let upc = emu.reg_read(RegisterARM64::PC).unwrap();
            assert!(step == 0 || upc == pc, "pc diverged before step {step}: uni {upc:#x} jit {pc:#x}");
            emu.emu_start(pc, u64::MAX, 0, 1).unwrap();
            let out = jit.run_bounded(&mut ctx, &mut Fetch, 1).unwrap();
            for (a, l) in std::mem::take(&mut g.invalidations) {
                jit.invalidate_range(a, l);
            }
            let _ = out;
            let mut bad = String::new();
            for i in 0..31 {
                let u = emu.reg_read(xreg(i)).unwrap();
                if u != ctx.x[i] {
                    bad += &format!(" x{i}: uni {u:#x} jit {:#x};", ctx.x[i]);
                }
            }
            let un = ((emu.reg_read(RegisterARM64::NZCV).unwrap() >> 28) & 0xF) as u8;
            if un != ctx.nzcv & 0xF {
                bad += &format!(" nzcv uni {un:04b} jit {:04b};", ctx.nzcv);
            }
            if !bad.is_empty() {
                panic!("step {step} pc {pc:#x} insn {word:08x}:{bad}");
            }
        }
    }
}

/// Investigation helper: prints which configurations disagree with the
/// baseline for `RUSTARMIC_CF_SEED` (`RUSTARMIC_CF_LSE=0`, `RUSTARMIC_CF_SMC=1`).
#[test]
#[ignore = "investigation helper"]
fn config_matrix_for_seed() {
    let seed: u64 = std::env::var("RUSTARMIC_CF_SEED").unwrap().parse().unwrap();
    let lse = std::env::var("RUSTARMIC_CF_LSE").map_or(true, |v| v != "0");
    let smc = std::env::var("RUSTARMIC_CF_SMC").is_ok();
    let p = gen_program(seed, GenOpts { lse, smc });
    let cfgs = configs();
    let base = run_rustarmic(&p, cfgs[0]);
    for cfg in &cfgs {
        let o = run_rustarmic(&p, *cfg);
        eprintln!("{cfg:?}: retired {}{}", o.retired, diff(&base, &o));
    }
}

/// Investigation helper: runs `RUSTARMIC_CF_SEED` in slices of
/// `RUSTARMIC_CF_BUDGET` next to a single-stepping reference and reports
/// the first slice after which the register state differs.
#[test]
#[ignore = "investigation helper"]
fn first_divergent_slice() {
    let seed: u64 = std::env::var("RUSTARMIC_CF_SEED").unwrap().parse().unwrap();
    let budget: u64 = std::env::var("RUSTARMIC_CF_BUDGET").map_or(3, |v| v.parse().unwrap());
    let lse = std::env::var("RUSTARMIC_CF_LSE").map_or(true, |v| v != "0");
    let p = gen_program(seed, GenOpts { lse, smc: false });
    let mk = |g: &mut Guest| {
        let mut ctx = CpuContext::default();
        ctx.mem_read = hk_read;
        ctx.mem_write = hk_write;
        ctx.mem_atomic = hk_atomic;
        ctx.mem_base = g.mem.as_mut_ptr();
        ctx.pc = CODE_BASE;
        ctx.sp = STACK_TOP;
        ctx.x = p.init_x;
        ctx.nzcv = p.init_nzcv;
        for i in 0..8 {
            ctx.v[i] = p.init_v[i];
        }
        ctx
    };
    let mut gr = Box::new(Guest { mem: image(&p), invalidations: Vec::new() });
    let mut gt = Box::new(Guest { mem: image(&p), invalidations: Vec::new() });
    let mut rctx = mk(&mut gr);
    let mut tctx = mk(&mut gt);
    let cfg = |mb| {
        let mut c = JitConfig { code_cache_bytes: 4 << 20, link_blocks: false, ..JitConfig::default() };
        c.translate.multiblock = mb;
        c
    };
    let mut rjit = Jit::new(cfg(false)).unwrap();
    let mut tjit = Jit::new(cfg(false)).unwrap();
    loop {
        let start = tctx.pc;
        GUEST.with(|s| *s.borrow_mut() = Some(&mut *gt as *mut Guest));
        let out = tjit.run_bounded(&mut tctx, &mut Fetch, budget).unwrap();
        GUEST.with(|s| *s.borrow_mut() = Some(&mut *gr as *mut Guest));
        for _ in 0..out.retired {
            rjit.run_bounded(&mut rctx, &mut Fetch, 1).unwrap();
        }
        if rctx.x != tctx.x || rctx.nzcv != tctx.nzcv || rctx.pc != tctx.pc {
            let at = ((start - CODE_BASE) / 4) as usize;
            let mut d = String::new();
            for i in 0..31 {
                if rctx.x[i] != tctx.x[i] {
                    d += &format!(" x{i}: ref {:#x} slice {:#x};", rctx.x[i], tctx.x[i]);
                }
            }
            panic!(
                "slice at {start:#x} retired {} -> pc ref {:#x} slice {:#x}, nzcv {:04b}/{:04b};{d}\ncode: {:08x?}",
                out.retired,
                rctx.pc,
                tctx.pc,
                rctx.nzcv,
                tctx.nzcv,
                &p.words[at..(at + 12).min(p.words.len())]
            );
        }
        if !matches!(out.reason, ExitReason::BudgetExhausted) {
            eprintln!("no divergence");
            return;
        }
    }
}

/// The generator really mixes in structure loads / stores, LDNP / STNP,
/// LDTR / STTR and helper-lowered instructions.
#[test]
fn generator_emits_simd_memory_and_helper_ops() {
    let (mut structure, mut nap, mut unpriv, mut helper) = (0, 0, 0, 0);
    for seed in 0..50 {
        let p = gen_program(seed, GenOpts { lse: true, smc: false });
        for &w in &p.words {
            if w & 0xBE00_0000 == 0x0C00_0000 {
                structure += 1;
            } else if w & 0x3B80_0000 == 0x2800_0000 {
                nap += 1;
            } else if w & 0x3B20_0C00 == 0x3800_0800 {
                unpriv += 1;
            } else if rustarmic::interp::plan(w).is_some() {
                helper += 1;
            }
        }
    }
    assert!(structure > 20 && nap > 5 && unpriv > 5 && helper > 20, "{structure} {nap} {unpriv} {helper}");
}
