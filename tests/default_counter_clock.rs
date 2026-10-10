use rustarmic::CpuContext;

#[test]
fn default_callback_and_read_only_accessor_share_one_epoch_across_contexts() {
    let mut contexts = [CpuContext::default(), CpuContext::default()];
    let mut previous = CpuContext::default_cntpct();
    for context in &mut contexts {
        assert_eq!(context.cntfrq_el0, 19_200_000);
        context.x[19] = 0xdead_beef;
        context.io_value = [0x1234, 0x5678];
        for _ in 0..64 {
            let before = CpuContext::default_cntpct();
            let value = unsafe { (context.read_cntpct)(context) };
            let after = CpuContext::default_cntpct();
            assert!(previous <= before && before <= value && value <= after);
            assert_eq!(context.x[19], 0xdead_beef);
            assert_eq!(context.io_value, [0x1234, 0x5678]);
            previous = after;
        }
    }
}

unsafe extern "C" fn custom_counter(context: *mut CpuContext) -> u64 {
    let context = unsafe { &mut *context };
    context.x[19] += 1;
    0xfeed_cafe + context.x[19]
}

#[test]
fn read_only_default_accessor_does_not_invoke_or_replace_custom_callbacks() {
    let mut context = CpuContext::default();
    context.read_cntpct = custom_counter;
    context.cntfrq_el0 = 12_345_678;
    let before = CpuContext::default_cntpct();
    let after = CpuContext::default_cntpct();
    assert!(before <= after);
    assert_eq!(context.x[19], 0);
    assert_eq!(context.cntfrq_el0, 12_345_678);
    // The custom callback remains opt-in, has the original mutable pointer,
    // and can still implement a counter from an unrelated embedder clock.
    let custom = unsafe { (context.read_cntpct)(&mut context) };
    assert_eq!(custom, 0xfeed_caff);
    assert_eq!(context.x[19], 1);
}
