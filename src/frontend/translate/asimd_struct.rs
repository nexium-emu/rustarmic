use disarm64::decoder::{ASISDLSE, ASISDLSEP, ASISDLSO, ASISDLSOP};

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

pub fn translate_multiple(em: &mut IrEmitter<'_>, insn: ASISDLSE) -> Result<InstStatus> {
    use ASISDLSE::*;
    let raw = match insn {
        ST1_LVt_SIMD_ADDR_SIMPLE(i) => i.0,
        ST2_LVt_SIMD_ADDR_SIMPLE(i) => i.0,
        ST3_LVt_SIMD_ADDR_SIMPLE(i) => i.0,
        ST4_LVt_SIMD_ADDR_SIMPLE(i) => i.0,
        LD1_LVt_SIMD_ADDR_SIMPLE(i) => i.0,
        LD2_LVt_SIMD_ADDR_SIMPLE(i) => i.0,
        LD3_LVt_SIMD_ADDR_SIMPLE(i) => i.0,
        LD4_LVt_SIMD_ADDR_SIMPLE(i) => i.0,
    };
    transfer_multiple(em, raw, false)
}

pub fn translate_multiple_post(em: &mut IrEmitter<'_>, insn: ASISDLSEP) -> Result<InstStatus> {
    use ASISDLSEP::*;
    let raw = match insn {
        ST1_LVt_SIMD_ADDR_POST(i) => i.0,
        ST2_LVt_SIMD_ADDR_POST(i) => i.0,
        ST3_LVt_SIMD_ADDR_POST(i) => i.0,
        ST4_LVt_SIMD_ADDR_POST(i) => i.0,
        LD1_LVt_SIMD_ADDR_POST(i) => i.0,
        LD2_LVt_SIMD_ADDR_POST(i) => i.0,
        LD3_LVt_SIMD_ADDR_POST(i) => i.0,
        LD4_LVt_SIMD_ADDR_POST(i) => i.0,
    };
    transfer_multiple(em, raw, true)
}

fn transfer_multiple(em: &mut IrEmitter<'_>, raw: u32, post_index: bool) -> Result<InstStatus> {
    let load = bit(raw, 22) != 0;
    // Derive the structure from raw opcode bits, not the generated decoder
    // variant: disarm64 labels Wonder's LD2 .2S (0x0c408aa0) as LD4.
    let (repeats, structure_elements) = match bits(raw, 12, 4) {
        0b0000 => (1, 4), // LD/ST4, lane-interleaved.
        0b0010 => (4, 1), // LD/ST1, four contiguous registers.
        0b0100 => (1, 3), // LD/ST3, lane-interleaved.
        0b0110 => (3, 1), // LD/ST1, three contiguous registers.
        0b0111 => (1, 1), // LD/ST1, one register.
        0b1000 => (1, 2), // LD/ST2, lane-interleaved.
        0b1010 => (2, 1), // LD/ST1, two contiguous registers.
        _ => {
            return Err(Error::Decode {
                pc: em.current_pc,
                opcode: raw,
            });
        }
    };
    let q = bit(raw, 30);
    let size = bits(raw, 10, 2);
    // The .1D arrangement is reserved for all interleaved structures.
    if q == 0 && size == 3 && structure_elements != 1 {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let vector_bytes = 8 << q;
    let element_bytes = 1 << size;
    let lanes = vector_bytes / element_bytes;
    let register_count = repeats * structure_elements;
    let rn = bits(raw, 5, 5) as u8;
    let rt = bits(raw, 0, 5) as u8;
    let base = em.get_x_or_sp(rn, true);
    let writeback = if post_index {
        let rm = bits(raw, 16, 5) as u8;
        let offset = if rm == 31 {
            em.const_u64((vector_bytes * register_count) as u64)
        } else {
            em.get_gpr(rm, RegSize::X)
        };
        Some(em.add(base, offset, RegSize::X))
    } else {
        None
    };
    // Stores snapshot their sources. Loads start with zeroed complete vectors,
    // which clears Q0's upper64 rather than retaining stale destination lanes.
    // Emit destination updates after all exact-width reads. Callback faults
    // use the backend's existing block-boundary reporting, not transactional
    // rollback or precise per-element aborts.
    let mut vectors: Vec<_> = if load {
        let zero = em.const_u64(0);
        let empty = em.vec_build_q(zero, zero);
        vec![empty; register_count as usize]
    } else {
        (0..register_count)
            .map(|i| em.get_v_q(((rt as u32 + i) & 31) as u8))
            .collect()
    };
    let mut offset = 0;
    for repeat in 0..repeats {
        for lane in 0..lanes {
            for structure in 0..structure_elements {
                let index = (repeat + structure) as usize;
                let vector = vectors[index];
                let address = if offset == 0 {
                    base
                } else {
                    let displacement = em.const_u64(offset as u64);
                    em.add(base, displacement, RegSize::X)
                };
                if load {
                    let value = em.load(address, element_bytes);
                    vectors[index] = em.vec_ins_gpr(vector, value, size, lane, true);
                } else {
                    let value = match element_bytes {
                        1 => em.vec_extract_u8(vector, lane),
                        2 => em.vec_extract_u16(vector, lane),
                        4 => em.vec_extract_u32(vector, lane),
                        8 if lane == 0 => em.vec_extract_lo64(vector),
                        8 => em.vec_extract_hi64(vector),
                        _ => unreachable!(),
                    };
                    em.store(address, value, element_bytes);
                }
                offset += element_bytes;
            }
        }
    }
    if load {
        for (i, vector) in vectors.into_iter().enumerate() {
            em.set_v_q(((rt as usize + i) & 31) as u8, vector);
        }
    }
    if let Some(writeback) = writeback {
        em.set_x_or_sp(rn, writeback, true);
    }
    Ok(InstStatus::Continue)
}

pub fn translate_single(em: &mut IrEmitter<'_>, insn: ASISDLSO) -> Result<InstStatus> {
    match insn {
        ASISDLSO::ST1_LEt_SIMD_ADDR_SIMPLE(i) => store_lane(em, i.0, false),
        ASISDLSO::LD1_LEt_SIMD_ADDR_SIMPLE(i) => load_lane(em, i.0, false),
        ASISDLSO::LD1R_LVt_AL_SIMD_ADDR_SIMPLE(i) => load_replicate(em, i.0, false),
        ASISDLSO::LD2R_LVt_AL_SIMD_ADDR_SIMPLE(i) => load_replicate(em, i.0, false),
        ASISDLSO::LD3R_LVt_AL_SIMD_ADDR_SIMPLE(i) => load_replicate(em, i.0, false),
        ASISDLSO::LD4R_LVt_AL_SIMD_ADDR_SIMPLE(i) => load_replicate(em, i.0, false),
        _ => Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: 0,
        }),
    }
}

pub fn translate_single_post(em: &mut IrEmitter<'_>, insn: ASISDLSOP) -> Result<InstStatus> {
    match insn {
        ASISDLSOP::ST1_LEt_SIMD_ADDR_POST(i) => store_lane(em, i.0, true),
        ASISDLSOP::LD1_LEt_SIMD_ADDR_POST(i) => load_lane(em, i.0, true),
        ASISDLSOP::LD1R_LVt_AL_SIMD_ADDR_POST(i) => load_replicate(em, i.0, true),
        ASISDLSOP::LD2R_LVt_AL_SIMD_ADDR_POST(i) => load_replicate(em, i.0, true),
        ASISDLSOP::LD3R_LVt_AL_SIMD_ADDR_POST(i) => load_replicate(em, i.0, true),
        ASISDLSOP::LD4R_LVt_AL_SIMD_ADDR_POST(i) => load_replicate(em, i.0, true),
        _ => Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: 0,
        }),
    }
}

fn lane_shape(em: &IrEmitter<'_>, raw: u32) -> Result<(u32, u32)> {
    let q = bit(raw, 30);
    let s = bit(raw, 12);
    let size = bits(raw, 10, 2);
    // Q selects the upper half of the lane index. It does not change the
    // transfer width: ST1/LD1 of one lane always accesses exactly one element.
    let (bytes, lane) = match bits(raw, 14, 2) {
        0 => (1, (q << 3) | (s << 2) | size),
        1 if size & 1 == 0 => (2, (q << 2) | (s << 1) | (size >> 1)),
        2 if size == 0 => (4, (q << 1) | s),
        2 if size == 1 && s == 0 => (8, q),
        _ => {
            return Err(Error::Decode {
                pc: em.current_pc,
                opcode: raw,
            });
        }
    };
    Ok((bytes, lane))
}

fn load_replicate(em: &mut IrEmitter<'_>, raw: u32, post_index: bool) -> Result<InstStatus> {
    // S is reserved for replicate loads; the no-offset form also reserves Rm.
    if bit(raw, 12) != 0 || (!post_index && bits(raw, 16, 5) != 0) {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let q = bit(raw, 30) != 0;
    let size = bits(raw, 10, 2);
    let bytes = 1 << size;
    let count = ((bit(raw, 13) << 1) | bit(raw, 21)) + 1;
    let rn = bits(raw, 5, 5) as u8;
    let rt = bits(raw, 0, 5) as u8;
    let base = em.get_x_or_sp(rn, true);
    // Snapshot the old register offset even when Rm == Rn. Immediate post-index
    // advances by the structure's memory footprint, not replicated vector size.
    let writeback = if post_index {
        let rm = bits(raw, 16, 5) as u8;
        let offset = if rm == 31 {
            em.const_u64((bytes * count) as u64)
        } else {
            em.get_gpr(rm, RegSize::X)
        };
        Some(em.add(base, offset, RegSize::X))
    } else {
        None
    };
    let mut vectors = Vec::with_capacity(count as usize);
    for index in 0..count {
        let address = if index == 0 {
            base
        } else {
            let offset = em.const_u64((index * bytes) as u64);
            em.add(base, offset, RegSize::X)
        };
        // Each register gets one exact-width read, replicated into its active
        // lanes. VecDupGpr clears the upper 64 bits for Q0, including .1D.
        let element = em.load(address, bytes);
        vectors.push(em.vec_dup_gpr(element, size, q));
    }
    // Keep the callback backend's existing block-boundary fault semantics.
    // These delayed updates are not a promise of transactional fault rollback.
    for (index, vector) in vectors.into_iter().enumerate() {
        em.set_v_q(((rt as usize + index) & 31) as u8, vector);
    }
    if let Some(writeback) = writeback {
        em.set_x_or_sp(rn, writeback, true);
    }
    Ok(InstStatus::Continue)
}

fn store_lane(em: &mut IrEmitter<'_>, raw: u32, post_index: bool) -> Result<InstStatus> {
    let (bytes, lane) = lane_shape(em, raw)?;
    let rn = bits(raw, 5, 5) as u8;
    let rt = bits(raw, 0, 5) as u8;
    let base = em.get_x_or_sp(rn, true);
    // Capture the old offset register before writeback, including Rm == Rn.
    let writeback = if post_index {
        let rm = bits(raw, 16, 5) as u8;
        let offset = if rm == 31 {
            em.const_u64(bytes as u64)
        } else {
            em.get_gpr(rm, RegSize::X)
        };
        Some(em.add(base, offset, RegSize::X))
    } else {
        None
    };
    let vector = em.get_v_q(rt);
    let value = match bytes {
        1 => em.vec_extract_u8(vector, lane),
        2 => em.vec_extract_u16(vector, lane),
        4 => em.vec_extract_u32(vector, lane),
        8 if lane == 0 => em.vec_extract_lo64(vector),
        8 => em.vec_extract_hi64(vector),
        _ => unreachable!(),
    };
    em.store(base, value, bytes);
    if let Some(writeback) = writeback {
        em.set_x_or_sp(rn, writeback, true);
    }
    Ok(InstStatus::Continue)
}

fn load_lane(em: &mut IrEmitter<'_>, raw: u32, post_index: bool) -> Result<InstStatus> {
    let (bytes, lane) = lane_shape(em, raw)?;
    let rn = bits(raw, 5, 5) as u8;
    let rt = bits(raw, 0, 5) as u8;
    let base = em.get_x_or_sp(rn, true);
    // Snapshot both base and register offset before any callback or writeback,
    // including Rm == Rn. Register 31 as Rm selects the immediate element size.
    let writeback = if post_index {
        let rm = bits(raw, 16, 5) as u8;
        let offset = if rm == 31 {
            em.const_u64(bytes as u64)
        } else {
            em.get_gpr(rm, RegSize::X)
        };
        Some(em.add(base, offset, RegSize::X))
    } else {
        None
    };
    let previous = em.get_v_q(rt);
    let value = em.load(base, bytes);
    // Q encodes part of the lane index, not a 64-bit destination width. Even
    // low-lane loads must preserve the rest of the complete 128-bit register.
    let result = em.vec_ins_gpr(previous, value, bytes.trailing_zeros(), lane, true);
    em.set_v_q(rt, result);
    if let Some(writeback) = writeback {
        em.set_x_or_sp(rn, writeback, true);
    }
    Ok(InstStatus::Continue)
}
