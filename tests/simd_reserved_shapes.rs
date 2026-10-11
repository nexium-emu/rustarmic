use disarm64::decoder::{
    ASIMDSHF, SHL_Vd_Vn_IMM_VLSL, SHRN_Vd_Vn_IMM_VLSR, SHRN2_Vd_Vn_IMM_VLSR,
    SSHR_Vd_Vn_IMM_VLSR, USHR_Vd_Vn_IMM_VLSR,
};
use rustarmic::Error;
use rustarmic::frontend::translate::asimd_shf;
use rustarmic::frontend::translator::{InstStatus, translate_instruction};
use rustarmic::ir::{Block, IrEmitter, Op};

const PC: u64 = 0x1000;

fn check_failure(error: Error, raw: u32, zero_immh: bool) {
    match error {
        Error::Decode { pc, opcode } if zero_immh => {
            assert_eq!((pc, opcode), (PC, raw));
        }
        Error::Unsupported { pc, opcode } if !zero_immh => {
            assert_eq!((pc, opcode), (PC, raw));
        }
        other => panic!("unexpected error for {raw:08x}: {other:?}"),
    }
}

#[test]
fn all_shrn_immediate_fields_reject_reserved_128_bit_source_shapes() {
    for upper in [false, true] {
        for immh in 0..16u32 {
            for immb in 0..8u32 {
                let raw = 0x0f00_8400 | (u32::from(upper) << 30)
                    | (immh << 19) | (immb << 16) | (31 << 5) | 31;
                let instruction = if upper {
                    ASIMDSHF::SHRN2_Vd_Vn_IMM_VLSR(SHRN2_Vd_Vn_IMM_VLSR(raw))
                } else {
                    ASIMDSHF::SHRN_Vd_Vn_IMM_VLSR(SHRN_Vd_Vn_IMM_VLSR(raw))
                };
                let mut block = Block::new(PC);
                let result = asimd_shf::translate(&mut IrEmitter::new(&mut block, PC), instruction);
                if immh == 0 || immh & 8 != 0 {
                    check_failure(result.unwrap_err(), raw, immh == 0);
                    assert_eq!(block.len(), 1, "{raw:08x}");
                } else {
                    assert_eq!(result.unwrap(), InstStatus::Continue, "{raw:08x}");
                    let source_size = 32 - immh.leading_zeros();
                    let width = 8u32 << (source_size - 1);
                    let shift = 2 * width - ((immh << 3) | immb);
                    let shifted = block.iter_live().find(|(_, a)| matches!(a.op,
                        Op::VecUshrImm16 | Op::VecUshrImm32 | Op::VecUshrImm64,
                    )).unwrap().1;
                    assert_eq!(shifted.op.size_log2(), source_size, "{raw:08x}");
                    assert_eq!(shifted.imm, ((shift as u64) << 1) | 1, "{raw:08x}");
                    assert_eq!(block.iter_live().filter(|(_, a)| a.op == Op::SetV).count(), 1);
                }
            }
        }
    }
}

#[test]
fn all_plain_shift_immediates_reject_vector_1d_and_keep_legal_lane_sizes() {
    for kind in 0..3 {
        for q in [false, true] {
            for immh in 0..16u32 {
                for immb in 0..8u32 {
                    let base = [0x0f00_5400, 0x2f00_0400, 0x0f00_0400][kind];
                    let raw = base | (u32::from(q) << 30) | (immh << 19)
                        | (immb << 16) | (31 << 5) | 31;
                    let instruction = match kind {
                        0 => ASIMDSHF::SHL_Vd_Vn_IMM_VLSL(SHL_Vd_Vn_IMM_VLSL(raw)),
                        1 => ASIMDSHF::USHR_Vd_Vn_IMM_VLSR(USHR_Vd_Vn_IMM_VLSR(raw)),
                        _ => ASIMDSHF::SSHR_Vd_Vn_IMM_VLSR(SSHR_Vd_Vn_IMM_VLSR(raw)),
                    };
                    let mut block = Block::new(PC);
                    let result = asimd_shf::translate(
                        &mut IrEmitter::new(&mut block, PC), instruction,
                    );
                    if immh == 0 || (immh & 8 != 0 && !q) {
                        check_failure(result.unwrap_err(), raw, immh == 0);
                        assert_eq!(block.len(), 1, "{raw:08x}");
                    } else {
                        assert_eq!(result.unwrap(), InstStatus::Continue, "{raw:08x}");
                        let size = 31 - immh.leading_zeros();
                        let width = 8u32 << size;
                        let immediate = (immh << 3) | immb;
                        let shift = if kind == 0 { immediate - width } else { 2 * width - immediate };
                        let family = [Op::VecShlImm8, Op::VecUshrImm8, Op::VecSshrImm8][kind];
                        let shifted = block.iter_live().find(|(_, a)| a.op.base() == family.base()).unwrap().1;
                        assert_eq!(shifted.op.size_log2(), size, "{raw:08x}");
                        assert_eq!(shifted.imm, ((shift as u64) << 1) | u64::from(q), "{raw:08x}");
                    }
                }
            }
        }
    }
}

#[test]
fn decoder_accepted_reserved_shrn_words_return_unsupported_before_any_ir() {
    for raw in [0x0f40_8400, 0x4f40_8400, 0x0f7f_87ff, 0x4f7f_87ff] {
        let mut block = Block::new(PC);
        let result = translate_instruction(&mut IrEmitter::new(&mut block, PC), raw);
        check_failure(result.unwrap_err(), raw, false);
        assert_eq!(block.len(), 1);
    }
}
