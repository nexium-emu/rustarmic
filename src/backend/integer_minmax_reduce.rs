/// A pure integer helper: no CpuContext, FP state, or guest flags are touched.
/// mode: signed bit 0, maximum bit 1, element log2 bytes bits 2..3, Q bit 4.
pub(super) extern "C" fn integer_minmax_reduce(low: u64, high: u64, mode: u64) -> u64 {
    let signed = mode & 1 != 0;
    let maximum = mode & 2 != 0;
    let size = ((mode >> 2) & 3) as u32;
    let q = mode & 16 != 0;
    debug_assert!(size <= 2 && (q || size < 2));
    let width = 8u32 << size;
    let mask = (1u64 << width) - 1;
    let lanes_per_half = 64 / width;
    let lanes = (if q { 128 } else { 64 }) / width;
    let key = |value: u64| -> i64 {
        if signed {
            ((value << (64 - width)) as i64) >> (64 - width)
        } else {
            value as i64
        }
    };
    let mut selected = low & mask;
    for lane in 1..lanes {
        let half = if lane < lanes_per_half { low } else { high };
        let shift = (lane % lanes_per_half) * width;
        let candidate = (half >> shift) & mask;
        let better = if maximum {
            key(candidate) > key(selected)
        } else {
            key(candidate) < key(selected)
        };
        if better {
            selected = candidate;
        }
    }
    selected
}
