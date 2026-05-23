use rustarmic::CpuContext;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_read8(_:u64, addr:u64, _:u64, _:*mut CpuContext) -> u8 {
    panic!("rustarmic: rustarmic_mem_read8 not implemented (addr={:#x})", addr);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_read16(_:u64, addr:u64, _:u64, _:*mut CpuContext) -> u16 {
    panic!("rustarmic: rustarmic_mem_read16 not implemented (addr={:#x})", addr);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_read32(_:u64, addr:u64, _:u64, _:*mut CpuContext) -> u32 {
    panic!("rustarmic: rustarmic_mem_read32 not implemented (addr={:#x})", addr);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_read64(_:u64, addr:u64, _:u64, _:*mut CpuContext) -> u64 {
    panic!("rustarmic: rustarmic_mem_read64 not implemented (addr={:#x})", addr);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_write8(_:u64, addr:u64, _:u8, _:*mut CpuContext) {
    panic!("rustarmic: rustarmic_mem_write8 not implemented (addr={:#x})", addr);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_write16(_:u64, addr:u64, _:u16, _:*mut CpuContext) {
    panic!("rustarmic: rustarmic_mem_write16 not implemented (addr={:#x})", addr);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_write32(_:u64, addr:u64, _:u32, _:*mut CpuContext) {
    panic!("rustarmic: rustarmic_mem_write32 not implemented (addr={:#x})", addr);
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_write64(_:u64, addr:u64, _:u64, _:*mut CpuContext) {
    panic!("rustarmic: rustarmic_mem_write64 not implemented (addr={:#x})", addr);
}
