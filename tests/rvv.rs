//! Run with RUSTFFT_RVV_TEST_EXPECT=available (or unavailable) to assert the intended CPU path.
use rustfft::FftPlannerRvv;

#[test]
fn availability() {
    let available = FftPlannerRvv::<f32>::new().is_ok();
    assert_eq!(FftPlannerRvv::<f64>::new().is_ok(), available);
    assert!(FftPlannerRvv::<i32>::new().is_err());
    if !cfg!(all(
        target_arch = "riscv64",
        target_os = "linux",
        feature = "rvv"
    )) {
        assert!(!available);
    }
    if let Ok(expected) = std::env::var("RUSTFFT_RVV_TEST_EXPECT") {
        match expected.as_str() {
            "available" => assert!(available, "RVV was required but was not detected"),
            "unavailable" => assert!(!available, "expected scalar fallback but RVV was detected"),
            _ => panic!("RUSTFFT_RVV_TEST_EXPECT must be available or unavailable"),
        }
    }
}

#[cfg(all(target_arch = "riscv64", target_os = "linux", feature = "rvv"))]
mod correctness {
    use num_traits::Float;
    use rustfft::{
        num_complex::Complex, Fft, FftDirection, FftNum, FftPlanner, FftPlannerRvv,
        FftPlannerScalar,
    };
    use std::{
        panic::{catch_unwind, AssertUnwindSafe},
        sync::Arc,
    };

    fn signal<T: FftNum>(count: usize) -> Vec<Complex<T>> {
        (0..count)
            .map(|i| {
                Complex::new(
                    T::from_f64(((i * 17 % 101) as f64 - 50.0) / 101.0).unwrap(),
                    T::from_f64(((i * 13 % 97) as f64 - 48.0) / 97.0).unwrap(),
                )
            })
            .collect()
    }

    fn check_close<T: FftNum + Float>(
        actual: &[Complex<T>],
        expected: &[Complex<T>],
        tolerance: T,
    ) {
        assert_eq!(actual.len(), expected.len());
        let mut error = T::zero();
        let mut magnitude = T::zero();
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(a.re.is_finite() && a.im.is_finite() && b.re.is_finite() && b.im.is_finite());
            error = error + (a - b).norm();
            magnitude = magnitude + b.norm();
        }
        assert!(
            error <= tolerance * magnitude.max(T::one()),
            "error={error:?}, magnitude={magnitude:?}"
        );
    }

    fn check_modes<T: FftNum + Float>(
        fft: &dyn Fft<T>,
        input: &[Complex<T>],
        expected: &[Complex<T>],
        tolerance: T,
    ) {
        let zero = Complex::new(T::zero(), T::zero());
        let guard = Complex::new(T::from_f64(123.0).unwrap(), T::from_f64(-321.0).unwrap());
        let mut protected = vec![guard; input.len() + 2];
        protected[1..=input.len()].copy_from_slice(input);
        let needed = fft.get_inplace_scratch_len();
        let mut scratch = vec![guard; needed + 1];
        // Also checks that dirty and oversized scratch works, and nothing writes past the buffer.
        fft.process_with_scratch(&mut protected[1..=input.len()], &mut scratch);
        check_close(&protected[1..=input.len()], expected, tolerance);
        assert_eq!(protected[0], guard);
        assert_eq!(*protected.last().unwrap(), guard);
        assert_eq!(scratch[needed], guard);
        let mut output = vec![zero; input.len()];
        let mut mutable_input = input.to_vec();
        let mut scratch = vec![guard; fft.get_outofplace_scratch_len()];
        fft.process_outofplace_with_scratch(&mut mutable_input, &mut output, &mut scratch);
        check_close(&output, expected, tolerance);
        let immutable_input = input.to_vec();
        let mut scratch = vec![guard; fft.get_immutable_scratch_len()];
        fft.process_immutable_with_scratch(&immutable_input, &mut output, &mut scratch);
        check_close(&output, expected, tolerance);
        assert_eq!(immutable_input, input);
    }

    fn compare<T: FftNum + Float>(tolerance: T) {
        let mut rvv = FftPlannerRvv::<T>::new().ok();
        let mut automatic = FftPlanner::<T>::new();
        let mut scalar = FftPlannerScalar::<T>::new();
        let lengths = (0..=16)
            .map(|power| 1 << power)
            .chain([3, 5, 6, 63, 65, 360, 1009, 1031]);
        for len in lengths {
            for direction in [FftDirection::Forward, FftDirection::Inverse] {
                for batch in [1, 3] {
                    let input = signal::<T>(len * batch);
                    let mut expected = input.clone();
                    scalar.plan_fft(len, direction).process(&mut expected);
                    let fft = automatic.plan_fft(len, direction);
                    check_modes(&*fft, &input, &expected, tolerance);
                    if let Some(planner) = &mut rvv {
                        let fft = planner.plan_fft(len, direction);
                        assert_eq!(fft.len(), len);
                        assert_eq!(fft.fft_direction(), direction);
                        check_modes(&*fft, &input, &expected, tolerance);
                    }
                }
            }
        }
    }

    #[test]
    fn scalar_agreement_f32() {
        compare::<f32>(1e-5);
    }
    #[test]
    fn scalar_agreement_f64() {
        compare::<f64>(1e-13);
    }

    fn round_trip<T: FftNum + Float>(tolerance: T) {
        let Ok(mut planner) = FftPlannerRvv::<T>::new() else {
            return;
        };
        for len in [1, 16, 32, 64, 128, 256, 1024, 65536] {
            let forward = planner.plan_fft_forward(len);
            let inverse = planner.plan_fft_inverse(len);
            for input in [
                signal::<T>(len),
                vec![Complex::new(T::zero(), T::zero()); len],
                {
                    let mut impulse = vec![Complex::new(T::zero(), T::zero()); len];
                    impulse[0].re = T::one();
                    impulse
                },
            ] {
                let mut result = input.clone();
                forward.process(&mut result);
                inverse.process(&mut result);
                for x in &mut result {
                    *x = *x / T::from_usize(len).unwrap();
                }
                check_close(&result, &input, tolerance);
            }
        }
    }
    #[test]
    fn round_trips_f32() {
        round_trip::<f32>(1e-5);
    }
    #[test]
    fn round_trips_f64() {
        round_trip::<f64>(1e-13);
    }

    #[test]
    fn plan_lifetime_and_zero_length() {
        let Ok(mut planner) = FftPlannerRvv::<f32>::new() else {
            return;
        };
        let empty = planner.plan_fft_forward(0);
        empty.process(&mut []);
        empty.process_outofplace_with_scratch(&mut [], &mut [], &mut []);
        empty.process_immutable_with_scratch(&[], &mut [], &mut []);
        let a = planner.plan_fft_forward(64);
        let b = planner.plan_fft_forward(64);
        assert!(Arc::ptr_eq(&a, &b));
        drop(planner);
        let mut data = signal::<f32>(64);
        a.process(&mut data);
        assert!(data.iter().all(|x| x.re.is_finite() && x.im.is_finite()));
    }

    fn rejects_invalid<T: FftNum + Float>() {
        let Ok(mut planner) = FftPlannerRvv::<T>::new() else {
            return;
        };
        for len in [64, 128, 256] {
            let fft = planner.plan_fft_forward(len);
            for count in [len - 1, len + 1] {
                assert!(
                    catch_unwind(AssertUnwindSafe(|| fft.process(&mut signal::<T>(count))))
                        .is_err()
                );
                let mut input = signal::<T>(count);
                let mut output = signal::<T>(count);
                assert!(
                    catch_unwind(AssertUnwindSafe(|| fft.process_outofplace_with_scratch(
                        &mut input,
                        &mut output,
                        &mut []
                    )))
                    .is_err()
                );
                assert!(
                    catch_unwind(AssertUnwindSafe(|| fft.process_immutable_with_scratch(
                        &input,
                        &mut output,
                        &mut []
                    )))
                    .is_err()
                );
            }
            let mut input = signal::<T>(len);
            let mut output = signal::<T>(len + 1);
            assert!(
                catch_unwind(AssertUnwindSafe(|| fft.process_outofplace_with_scratch(
                    &mut input,
                    &mut output,
                    &mut []
                )))
                .is_err()
            );
            assert!(
                catch_unwind(AssertUnwindSafe(|| fft.process_immutable_with_scratch(
                    &input,
                    &mut output,
                    &mut []
                )))
                .is_err()
            );
            let mut scratch = signal::<T>(fft.get_inplace_scratch_len() - 1);
            assert!(catch_unwind(AssertUnwindSafe(
                || fft.process_with_scratch(&mut input, &mut scratch)
            ))
            .is_err());
        }
    }
    #[test]
    fn invalid_arguments_f32() {
        rejects_invalid::<f32>();
    }
    #[test]
    fn invalid_arguments_f64() {
        rejects_invalid::<f64>();
    }
}
