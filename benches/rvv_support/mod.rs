use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use rustfft::{num_complex::Complex, FftDirection, FftNum, FftPlannerRvv, FftPlannerScalar};
use std::hint::black_box;

pub fn run(c: &mut Criterion, compare_scalar: bool) {
    FftPlannerRvv::<f32>::new()
        .expect("RVV benchmarks require nightly, --features rvv, and RVV 1.0 on riscv64 Linux");
    #[cfg(all(target_arch = "riscv64", target_os = "linux", feature = "rvv"))]
    {
        let bytes: usize;
        // SAFETY: successful planner creation above verified vector support. Reading vlenb
        // does not change vector state; its value is used only to annotate benchmark output.
        unsafe { core::arch::asm!("csrr {}, 0xc22", out(reg) bytes, options(nomem, nostack)) };
        eprintln!("RVV detected; vector length = {} bits", bytes * 8);
    }
    run_precision::<f32>(c, "f32", compare_scalar);
    run_precision::<f64>(c, "f64", compare_scalar);
}

fn run_precision<T: FftNum>(c: &mut Criterion, precision: &str, compare_scalar: bool) {
    let mut rvv = FftPlannerRvv::<T>::new().unwrap();
    let mut scalar = FftPlannerScalar::<T>::new();
    let lengths: Vec<usize> = (4..=20).map(|exponent| 1 << exponent).collect();
    let fallbacks = if compare_scalar {
        &[63, 360, 1009][..]
    } else {
        &[]
    };
    for direction in [FftDirection::Forward, FftDirection::Inverse] {
        let direction_name = match direction {
            FftDirection::Forward => "forward",
            FftDirection::Inverse => "inverse",
        };
        let prefix = if compare_scalar {
            "rvvcomparison"
        } else {
            "rvv"
        };
        let mut group = c.benchmark_group(format!("{prefix}/{precision}/{direction_name}"));
        for &len in lengths.iter().chain(fallbacks) {
            for batch in [1, 10] {
                if batch == 10 && len > 1024 {
                    continue;
                }
                let count = len * batch;
                let input: Vec<_> = (0..count)
                    .map(|i| {
                        Complex::new(
                            T::from_f64(((i * 17 % 101) as f64 - 50.0) / 101.0).unwrap(),
                            T::from_f64(((i * 13 % 97) as f64 - 48.0) / 97.0).unwrap(),
                        )
                    })
                    .collect();
                let rvv_fft = rvv.plan_fft(len, direction);
                let mut implementations = vec![("rvv", rvv_fft)];
                if compare_scalar {
                    implementations.push(("scalar", scalar.plan_fft(len, direction)));
                }
                group.throughput(Throughput::Elements(count as u64));
                for (backend, fft) in implementations {
                    let mut scratch =
                        vec![Complex::new(T::zero(), T::zero()); fft.get_inplace_scratch_len()];
                    group.bench_with_input(
                        BenchmarkId::new(format!("{backend}/batch{batch}"), len),
                        &input,
                        |b, signal| {
                            // Reset outside timing; repeated unnormalized FFTs overflow. Limit
                            // batched allocations for the large cases on memory-constrained SBCs.
                            let batch_size =
                                BatchSize::NumIterations(if count <= 1024 { 16 } else { 1 });
                            b.iter_batched_ref(
                                || signal.clone(),
                                |buffer| {
                                    fft.process_with_scratch(
                                        black_box(buffer),
                                        black_box(&mut scratch),
                                    );
                                    black_box(buffer);
                                },
                                batch_size,
                            );
                        },
                    );
                }
            }
        }
        group.finish();
    }
}
