//! AdvSIMD structure loads and stores: LD1-LD4 / ST1-ST4 (multiple
//! structures), LD1-LD4 / ST1-ST4 (single structure, one lane) and
//! LD1R-LD4R (load and replicate), with and without post-index writeback.
//!
//! Memory is accessed in ascending address order through the ordinary
//! `Load*` / `Store*` IR ops, so every access takes the same inline
//! page-table fastmem or callback path as scalar loads and stores.  Whole
//! registers move as 8 / 16-byte accesses and are (de)interleaved in
//! registers with UZP / ZIP / TBL; lane forms access one element per
//! register.  A fault reports the first failing access (later callback
//! accesses of the block are skipped once `mem_fault` is set).

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{IrEmitter, ValueRef};
use crate::util::bits::{bit, bits};

fn unsupported(em: &IrEmitter<'_>, raw: u32) -> Error {
    Error::Unsupported {
        pc: em.current_pc,
        opcode: raw,
    }
}

/// Base address, and the post-index writeback once the transfer is done.
fn base_address(em: &mut IrEmitter<'_>, raw: u32) -> ValueRef {
    em.get_x_or_sp(bits(raw, 5, 5) as u8, true)
}

fn writeback(em: &mut IrEmitter<'_>, raw: u32, base: ValueRef, imm_bytes: u64) {
    let rn = bits(raw, 5, 5) as u8;
    let rm = bits(raw, 16, 5) as u8;
    // Rm = 31 encodes the immediate form: the offset is the transfer size.
    let off = if rm == 31 {
        em.const_u64(imm_bytes)
    } else {
        em.get_x(rm)
    };
    let next = em.add(base, off, RegSize::X);
    em.set_x_or_sp(rn, next, true);
}

fn addr_at(em: &mut IrEmitter<'_>, base: ValueRef, off: u64) -> ValueRef {
    if off == 0 {
        return base;
    }
    let c = em.const_u64(off);
    em.add(base, c, RegSize::X)
}

/// TBL index vector for gathering byte `i` of the result from
/// `src_byte(i)` (0xFF when the byte is not needed).
fn tbl_indices(em: &mut IrEmitter<'_>, src_byte: impl Fn(u32) -> u32) -> ValueRef {
    let mut lo = 0u64;
    let mut hi = 0u64;
    for i in 0..16u32 {
        let b = (src_byte(i) & 0xFF) as u64;
        if i < 8 {
            lo |= b << (8 * i);
        } else {
            hi |= b << (8 * (i - 8));
        }
    }
    em.const_u128(lo, hi)
}

/// LD1-LD4 / ST1-ST4 (multiple structures).
pub fn translate_multiple(em: &mut IrEmitter<'_>, raw: u32, post: bool) -> Result<InstStatus> {
    let q = bit(raw, 30) == 1;
    let load = bit(raw, 22) == 1;
    let opcode = bits(raw, 12, 4);
    let size = bits(raw, 10, 2);
    let rt = bits(raw, 0, 5);
    if post && bits(raw, 21, 1) != 0 || !post && bits(raw, 16, 6) != 0 {
        return Err(unsupported(em, raw));
    }
    let (rpt, selem) = match opcode {
        0b0000 => (1u32, 4u32),
        0b0010 => (4, 1),
        0b0100 => (1, 3),
        0b0110 => (3, 1),
        0b0111 => (1, 1),
        0b1000 => (1, 2),
        0b1010 => (2, 1),
        _ => return Err(unsupported(em, raw)),
    };
    if size == 0b11 && !q && selem != 1 {
        return Err(unsupported(em, raw));
    }
    let regbytes: u32 = if q { 16 } else { 8 };
    let nregs = rpt * selem;
    let total = regbytes * nregs;
    let reg = |i: u32| ((rt + i) % 32) as u8;

    let base = base_address(em, raw);

    if selem == 1 {
        // Consecutive registers, each a contiguous block of memory.
        if load {
            let mut vals = [ValueRef::NONE; 4];
            for (i, v) in vals.iter_mut().enumerate().take(nregs as usize) {
                let a = addr_at(em, base, (i as u32 * regbytes) as u64);
                *v = em.load_elements(a, regbytes, 1 << size);
            }
            for (i, &v) in vals.iter().enumerate().take(nregs as usize) {
                if q {
                    em.set_v_q(reg(i as u32), v);
                } else {
                    em.set_v_d(reg(i as u32), v);
                }
            }
        } else {
            for i in 0..nregs {
                let v = if q {
                    em.get_v_q(reg(i))
                } else {
                    em.get_v_d(reg(i))
                };
                let a = addr_at(em, base, (i * regbytes) as u64);
                em.store_elements(a, v, regbytes, 1 << size);
            }
        }
    } else if load {
        load_interleaved(em, base, selem, size, q, reg);
    } else {
        store_interleaved(em, base, selem, size, q, reg);
    }

    if post {
        writeback(em, raw, base, total as u64);
    }
    Ok(InstStatus::Continue)
}

/// LD2 / LD3 / LD4: `selem` registers whose element `e` of register `s`
/// is memory element `e * selem + s`.
fn load_interleaved(
    em: &mut IrEmitter<'_>,
    base: ValueRef,
    selem: u32,
    size: u32,
    q: bool,
    reg: impl Fn(u32) -> u8,
) {
    let eb = 1u32 << size;
    let total = (if q { 16 } else { 8 }) * selem;
    // Memory as 16-byte chunks (the last one may be 8 bytes).
    let mut chunks = [ValueRef::NONE; 4];
    let mut off = 0u32;
    let mut n = 0usize;
    while off < total {
        let a = addr_at(em, base, off as u64);
        chunks[n] = if total - off >= 16 {
            em.load_elements(a, 16, eb)
        } else {
            let d = em.load_elements(a, 8, eb);
            let z = em.const_u64(0);
            em.vec_build_q(d, z)
        };
        off += 16;
        n += 1;
    }
    let mut out = [ValueRef::NONE; 4];
    match selem {
        2 => {
            // Q: chunks c0 c1; D: one chunk holding both registers.
            let (c0, c1) = if q { (chunks[0], chunks[1]) } else { (chunks[0], chunks[0]) };
            out[0] = em.vec_uzp1(c0, c1, size, true);
            out[1] = em.vec_uzp2(c0, c1, size, true);
        }
        4 => {
            let (t0, t1, t2, t3);
            if q {
                t0 = em.vec_uzp1(chunks[0], chunks[1], size, true);
                t1 = em.vec_uzp2(chunks[0], chunks[1], size, true);
                t2 = em.vec_uzp1(chunks[2], chunks[3], size, true);
                t3 = em.vec_uzp2(chunks[2], chunks[3], size, true);
            } else {
                t0 = em.vec_uzp1(chunks[0], chunks[1], size, true);
                t1 = em.vec_uzp2(chunks[0], chunks[1], size, true);
                t2 = t0;
                t3 = t1;
            }
            out[0] = em.vec_uzp1(t0, t2, size, true);
            out[1] = em.vec_uzp1(t1, t3, size, true);
            out[2] = em.vec_uzp2(t0, t2, size, true);
            out[3] = em.vec_uzp2(t1, t3, size, true);
        }
        _ => {
            // LD3: gather with a table lookup over the loaded bytes.
            for s in 0..3u32 {
                let idx = tbl_indices(em, |b| {
                    let (e, k) = (b / eb, b % eb);
                    (e * 3 + s) * eb + k
                });
                out[s as usize] = if q {
                    em.vec_tbl3(chunks[0], chunks[1], chunks[2], idx, true)
                } else {
                    em.vec_tbl2(chunks[0], chunks[1], idx, true)
                };
            }
        }
    }
    for s in 0..selem {
        let v = out[s as usize];
        if q {
            em.set_v_q(reg(s), v);
        } else {
            let lo = em.vec_extract_lo64(v);
            em.set_v_d(reg(s), lo);
        }
    }
}

/// ST2 / ST3 / ST4: the inverse of [`load_interleaved`].
fn store_interleaved(
    em: &mut IrEmitter<'_>,
    base: ValueRef,
    selem: u32,
    size: u32,
    q: bool,
    reg: impl Fn(u32) -> u8,
) {
    let eb = 1u32 << size;
    let total = (if q { 16 } else { 8 }) * selem;
    let mut src = [ValueRef::NONE; 4];
    for s in 0..selem {
        src[s as usize] = em.get_v_q(reg(s));
    }
    let mut chunks = [ValueRef::NONE; 4];
    match selem {
        2 => {
            chunks[0] = em.vec_zip1(src[0], src[1], size, true);
            if q {
                chunks[1] = em.vec_zip2(src[0], src[1], size, true);
            }
        }
        4 => {
            let p = em.vec_zip1(src[0], src[2], size, true);
            let r = em.vec_zip1(src[1], src[3], size, true);
            chunks[0] = em.vec_zip1(p, r, size, true);
            chunks[1] = em.vec_zip2(p, r, size, true);
            if q {
                let p2 = em.vec_zip2(src[0], src[2], size, true);
                let r2 = em.vec_zip2(src[1], src[3], size, true);
                chunks[2] = em.vec_zip1(p2, r2, size, true);
                chunks[3] = em.vec_zip2(p2, r2, size, true);
            }
        }
        _ => {
            let nchunks = total.div_ceil(16);
            for j in 0..nchunks {
                let idx = tbl_indices(em, |i| {
                    let p = 16 * j + i;
                    let (g, k) = (p / eb, p % eb);
                    let (e, s) = (g / 3, g % 3);
                    16 * s + e * eb + k
                });
                chunks[j as usize] = em.vec_tbl3(src[0], src[1], src[2], idx, true);
            }
        }
    }
    let mut off = 0u32;
    let mut n = 0usize;
    while off < total {
        let a = addr_at(em, base, off as u64);
        if total - off >= 16 {
            em.store_elements(a, chunks[n], 16, eb);
        } else {
            let lo = em.vec_extract_lo64(chunks[n]);
            em.store_elements(a, lo, 8, eb);
        }
        off += 16;
        n += 1;
    }
}

/// LD1-LD4 / ST1-ST4 (single structure) and LD1R-LD4R.
pub fn translate_single(em: &mut IrEmitter<'_>, raw: u32, post: bool) -> Result<InstStatus> {
    let q = bit(raw, 30);
    let load = bit(raw, 22) == 1;
    let r = bit(raw, 21);
    let opcode = bits(raw, 13, 3);
    let s = bit(raw, 12);
    let size = bits(raw, 10, 2);
    let rt = bits(raw, 0, 5);
    if !post && bits(raw, 16, 5) != 0 {
        return Err(unsupported(em, raw));
    }
    let selem = (((opcode & 1) << 1) | r) + 1;
    let mut scale = opcode >> 1;
    let mut replicate = false;
    let index = match scale {
        0 => (q << 3) | (s << 2) | size,
        1 => {
            if size & 1 != 0 {
                return Err(unsupported(em, raw));
            }
            (q << 2) | (s << 1) | (size >> 1)
        }
        2 => {
            if size & 2 != 0 {
                return Err(unsupported(em, raw));
            }
            if size & 1 == 0 {
                (q << 1) | s
            } else {
                if s != 0 {
                    return Err(unsupported(em, raw));
                }
                scale = 3;
                q
            }
        }
        _ => {
            if !load || s != 0 {
                return Err(unsupported(em, raw));
            }
            scale = size;
            replicate = true;
            0
        }
    };
    let eb = 1u32 << scale;
    let reg = |i: u32| ((rt + i) % 32) as u8;
    let base = base_address(em, raw);

    if replicate {
        let mut vals = [ValueRef::NONE; 4];
        for (i, v) in vals.iter_mut().enumerate().take(selem as usize) {
            let a = addr_at(em, base, (i as u32 * eb) as u64);
            *v = em.load(a, eb);
        }
        for i in 0..selem {
            let d = em.vec_dup_gpr(vals[i as usize], scale, q == 1);
            em.set_v_q(reg(i), d);
        }
    } else if load {
        let mut vals = [ValueRef::NONE; 4];
        for (i, v) in vals.iter_mut().enumerate().take(selem as usize) {
            let a = addr_at(em, base, (i as u32 * eb) as u64);
            *v = em.load(a, eb);
        }
        for i in 0..selem {
            let old = em.get_v_q(reg(i));
            let new = em.vec_ins_gpr(old, vals[i as usize], scale, index, true);
            em.set_v_q(reg(i), new);
        }
    } else {
        for i in 0..selem {
            let v = em.get_v_q(reg(i));
            let elem = match scale {
                0 => em.vec_extract_u8(v, index),
                1 => em.vec_extract_u16(v, index),
                2 => em.vec_extract_u32(v, index),
                _ => {
                    if index == 0 {
                        em.vec_extract_lo64(v)
                    } else {
                        em.vec_extract_hi64(v)
                    }
                }
            };
            let a = addr_at(em, base, (i * eb) as u64);
            em.store(a, elem, eb);
        }
    }

    if post {
        writeback(em, raw, base, (selem * eb) as u64);
    }
    Ok(InstStatus::Continue)
}
