//! Vector-length-agnostic RVV 1.0 power-of-two FFTs.
mod kernels;
mod radix4;

use std::{any::TypeId, sync::Arc};

use crate::{fft_cache::FftCache, Fft, FftDirection, FftNum, FftPlannerScalar};
use radix4::RvvRadix4;

/// Plans power-of-two FFTs using RVV 1.0 on riscv64 Linux.
///
/// Requires nightly Rust and the optional `rvv` feature. [`Self::new`] checks vector support
/// before any vector instructions execute. Transforms of length 64 and above that are powers
/// of two use RVV; smaller transforms and all other lengths use scalar algorithms.
/// Both f32 and f64 are supported, at any vector length supported by RVV 1.0.
/// No global vector target flags are needed.
///
/// Plans own their data and remain usable after the planner is dropped. Reusing a planner
/// shares cached plans of the same length and direction.
pub struct FftPlannerRvv<T: FftNum> {
    cache: FftCache<T>,
    scalar: FftPlannerScalar<T>,
}

impl<T: FftNum> FftPlannerRvv<T> {
    /// Returns `Err(())` if vector support is unavailable or T is neither f32 nor f64.
    pub fn new() -> Result<Self, ()> {
        if (TypeId::of::<T>() == TypeId::of::<f32>() || TypeId::of::<T>() == TypeId::of::<f64>())
            && std::arch::is_riscv_feature_detected!("v")
        {
            Ok(Self {
                cache: FftCache::new(),
                scalar: FftPlannerScalar::new(),
            })
        } else {
            Err(())
        }
    }

    /// Plans a transform, reusing cached plans with the same length and direction.
    pub fn plan_fft(&mut self, len: usize, direction: FftDirection) -> Arc<dyn Fft<T>> {
        if let Some(fft) = self.cache.get(len, direction) {
            return fft;
        }
        let fft: Arc<dyn Fft<T>> = if uses_rvv(len) {
            // SAFETY: new() checked CPU support and T. No other constructor is exposed.
            Arc::new(unsafe { RvvRadix4::new(len, direction) })
        } else {
            self.scalar.plan_fft(len, direction)
        };
        self.cache.insert(&fft);
        fft
    }

    /// Plans a forward transform.
    pub fn plan_fft_forward(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Forward)
    }

    /// Plans an inverse transform.
    pub fn plan_fft_inverse(&mut self, len: usize) -> Arc<dyn Fft<T>> {
        self.plan_fft(len, FftDirection::Inverse)
    }
}

fn uses_rvv(len: usize) -> bool {
    len >= 64 && len.is_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_and_cache() {
        let available = std::arch::is_riscv_feature_detected!("v");
        assert_eq!(FftPlannerRvv::<f32>::new().is_ok(), available);
        assert_eq!(FftPlannerRvv::<f64>::new().is_ok(), available);
        assert!(FftPlannerRvv::<i32>::new().is_err());
        let Ok(mut planner) = FftPlannerRvv::<f32>::new() else {
            return;
        };
        for len in [0, 1, 16, 32, 63, 64, 128, 256, 360, 1024, 1031] {
            let a = planner.plan_fft_forward(len);
            let b = planner.plan_fft_forward(len);
            assert!(Arc::ptr_eq(&a, &b));
            let inverse = planner.plan_fft_inverse(len);
            assert!(!Arc::ptr_eq(&a, &inverse));
            assert_eq!(a.fft_direction(), FftDirection::Forward);
            assert_eq!(inverse.fft_direction(), FftDirection::Inverse);
            // RVV uses one transform of scratch. Small plans retain the scalar contract.
            if len.is_power_of_two() {
                assert_eq!(
                    a.get_inplace_scratch_len(),
                    if uses_rvv(len) {
                        len
                    } else {
                        FftPlannerScalar::<f32>::new()
                            .plan_fft_forward(len)
                            .get_inplace_scratch_len()
                    }
                );
            }
        }
    }
}
