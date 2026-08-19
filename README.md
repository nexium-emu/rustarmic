# rustarmic

Rustarmic is an AArch64-to-x86-64 JIT with a bounded execution surface. New
embedders should use `Engine` and retain its `Arc<SharedRuntime>`; legacy
`Jit` remains available for low-level tests.

`EngineConfig` defaults to accurate FP behavior, a sparse/page-table memory
mode, a 256 MiB code cache, and 64-instruction maximum blocks. `Engine::run`
returns a retired count plus a structured `StopReason`; `step` uses a budget
of one and budget zero executes nothing. Hosts below x86-64 SSE4.1 are
rejected during construction. Dynarmic integration belongs in the consumer
and is an explicit manual oracle/fallback, never an automatic mid-run path.

