use std::{any::TypeId, sync::Arc};

use num_complex::Complex;

use super::kernels;
use crate::{
    algorithm::butterflies::{Butterfly16, Butterfly32},
    array_utils::reverse_bits,
    simd::simd_radixn::table_transpose,
    twiddles::compute_twiddle,
    Direction, Fft, FftDirection, FftNum, Length,
};

type Kernel = unsafe fn(*mut u8, *const u8, usize);

// Unlike SimdRadix4, twiddles are stored as scalars and no vector width is part of a plan.
// A plan may be shared across threads: every kernel invocation chooses its own VL.
pub(super) struct RvvRadix4<T: FftNum> {
    len: usize,
    direction: FftDirection,
    base: Arc<dyn Fft<T>>,
    reversed_columns: Box<[usize]>,
    twiddles: Box<[Complex<T>]>,
    kernel: Kernel,
}

impl<T: FftNum> RvvRadix4<T> {
    /// Safety: the machine must support V, and T must be f32 or f64.
    pub(super) unsafe fn new(len: usize, direction: FftDirection) -> Self {
        assert!(len >= 64 && len.is_power_of_two());
        let inverse = direction == FftDirection::Inverse;
        // Select once per plan, rather than dispatching on precision/direction in inner loops.
        let kernel: Kernel = if TypeId::of::<T>() == TypeId::of::<f32>() {
            if inverse {
                kernels::radix4_f32::<true>
            } else {
                kernels::radix4_f32::<false>
            }
        } else {
            assert_eq!(TypeId::of::<T>(), TypeId::of::<f64>());
            if inverse {
                kernels::radix4_f64::<true>
            } else {
                kernels::radix4_f64::<false>
            }
        };
        let base: Arc<dyn Fft<T>> = if len.trailing_zeros() % 2 == 0 {
            Arc::new(Butterfly16::new(direction))
        } else {
            Arc::new(Butterfly32::new(direction))
        };
        let width = len / base.len();
        let digits = width.trailing_zeros() / 2;
        let reversed_columns: Box<[_]> = (0..width).map(|x| reverse_bits::<4>(x, digits)).collect();
        // table_transpose uses unchecked indexing and requires this invariant.
        assert!(reversed_columns.iter().all(|&r| r < width));
        let mut twiddles = Vec::with_capacity(len - base.len());
        let mut columns = base.len();
        while columns < len {
            // Three contiguous complex streams for segmented vector loads, one per radix row.
            for row in 1..4 {
                for column in 0..columns {
                    twiddles.push(compute_twiddle(column * row, columns * 4, direction));
                }
            }
            columns *= 4;
        }
        Self {
            len,
            direction,
            base,
            reversed_columns,
            twiddles: twiddles.into_boxed_slice(),
            kernel,
        }
    }

    fn transpose(&self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        table_transpose::<_, 4>(self.base.len(), &self.reversed_columns, input, output);
    }

    fn cross_ffts(&self, output: &mut [Complex<T>]) {
        let mut columns = self.base.len();
        let mut twiddles = &self.twiddles[..];
        while columns < self.len {
            for block in output.chunks_exact_mut(columns * 4) {
                // SAFETY: the private constructor is reached only after detecting V and checking
                // T. Each block contains four full rows; this layer has three full twiddle rows.
                // The kernel uses scalar-aligned loads and bounds each access by its chosen VL.
                unsafe {
                    (self.kernel)(block.as_mut_ptr().cast(), twiddles.as_ptr().cast(), columns)
                };
            }
            twiddles = &twiddles[3 * columns..];
            columns *= 4;
        }
    }

    fn perform_fft_inplace(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let transposed = &mut scratch[..self.len];
        self.transpose(buffer, transposed);
        self.base
            .process_outofplace_with_scratch(transposed, buffer, &mut []);
        self.cross_ffts(buffer);
    }

    fn perform_fft_immut(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        _scratch: &mut [Complex<T>],
    ) {
        self.transpose(input, output);
        self.base.process_with_scratch(output, &mut []);
        self.cross_ffts(output);
    }

    fn perform_fft_out_of_place(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.perform_fft_immut(input, output, scratch);
    }
}

boilerplate_fft!(
    RvvRadix4,
    |this: &RvvRadix4<T>| this.len,
    |this: &RvvRadix4<T>| this.len,
    |_this: &RvvRadix4<T>| 0,
    |_this: &RvvRadix4<T>| 0
);
