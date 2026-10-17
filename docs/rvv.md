# RISC-V vector support

The optional `rvv` feature enables RVV 1.0 on **riscv64 Linux**, using nightly Rust.
`FftPlanner` detects vector support automatically. `FftPlannerRvv::new()` returns
`Err(())` when RVV is unavailable, allowing an application to choose its own fallback.

Both f32 and f64 power-of-two transforms of length 64 and above use vector radix-4
stages with scalar 16- or 32-point base butterflies. Other lengths use the scalar
planner. Forward/inverse, batching, and all three processing modes are supported.
Transforms remain unnormalized, just like the other backends.

Plans adapt to the hardware vector length during each kernel invocation. They do
not assume the Orange Pi RV2's vector width, and can be shared across threads.
This feature has no effect on other targets and does not change the stable Rust
requirement for builds without RVV.

No global `RUSTFLAGS` or `-C target-feature=+v` setting is needed. Keep the baseline
`riscv64gc-unknown-linux-gnu` target to retain runtime fallback on machines without
vectors. The OS must enable vector use and expose it through Rust's runtime
feature detection. If availability fails, inspect the kernel and vector support
rather than forcing vector instructions on globally. Older vendor kernels may
not expose the facilities Rust's detection needs.

## Orange Pi RV2: native testing

Codex does not need to run on the board: copy or check out this source tree there,
and run these commands from the repository root with an installed nightly Rust.
The initial development toolchain was nightly (rustc 1.101.0-nightly,
commit db8f076d2, dated 2026-10-03); record the toolchain actually used for each hardware run.

```sh
rustc +nightly -Vv
uname -a
lscpu

RUSTFFT_RVV_TEST_EXPECT=available cargo +nightly test --release \
  --no-default-features --features rvv

cargo +nightly bench --no-default-features --features rvv \
  --bench bench_rustfft_rvv
cargo +nightly bench --no-default-features --features rvv \
  --bench bench_compare_scalar_rvv
```

`RUSTFFT_RVV_TEST_EXPECT=available` makes tests fail if vector detection fails;
otherwise hardware-specific tests can skip when RVV is unavailable. The benchmark
also requires RVV and reports the detected vector length in bits. Preserve this
output along with `rustc -Vv`, kernel/CPU information, and Criterion's console
results and `target/criterion` directory.

For a quick comparison before running the full suite:

```sh
cargo +nightly bench --no-default-features --features rvv \
  --bench bench_compare_scalar_rvv -- 'rvvcomparison/f32/forward/.*/1024$'
```

Benchmarks cover both precisions and directions, sizes 16 through 1,048,576, and
batches of ten for small sizes. The comparison also covers scalar fallback lengths
63, 360, and 1009. Plan creation, input reset, allocation, and scratch preparation
are outside the timed execution. Pinning the process to one CPU with `taskset`,
keeping the board cool, and avoiding other workloads helps make comparisons
repeatable. Small transforms include scalar fallbacks intentionally.

Real hardware is required to assess speed. QEMU is used only for correctness;
no speedup is claimed from emulator timings. Hardware results should be checked
before changing thresholds or enabling this feature by default.

## Cross-compilation and QEMU

On Debian/Ubuntu, install `gcc-riscv64-linux-gnu` and `qemu-user`, then:

```sh
rustup target add --toolchain nightly riscv64gc-unknown-linux-gnu
export CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=riscv64-linux-gnu-gcc
export CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_RUNNER='qemu-riscv64 -L /usr/riscv64-linux-gnu'
export QEMU_CPU='rv64,v=true,vlen=256,elen=64,vext_spec=v1.0'

RUSTFFT_RVV_TEST_EXPECT=available cargo +nightly test --release \
  --target riscv64gc-unknown-linux-gnu --no-default-features --features rvv
cargo +nightly bench --no-run --target riscv64gc-unknown-linux-gnu \
  --no-default-features --features rvv --bench bench_rustfft_rvv \
  --bench bench_compare_scalar_rvv
```

Repeat tests with `vlen=128` and `vlen=512`. Kernel tests exercise partial vectors,
including column counts that are not powers of two. To verify runtime fallback
using the same executable code, rerun tests with:

```sh
export QEMU_CPU='rv64,v=false'
RUSTFFT_RVV_TEST_EXPECT=unavailable cargo +nightly test --release \
  --target riscv64gc-unknown-linux-gnu --no-default-features --features rvv
```

Local emulation uses QEMU 10.0.13. To build binaries for transfer to the board,
omit the runner variable and use `cargo +nightly test --no-run` or
`cargo +nightly bench --no-run` with the target and feature flags above. Executables
are printed by Cargo under `target/riscv64gc-unknown-linux-gnu/release`; the board's
userspace must be compatible with the cross-linker's libc.
